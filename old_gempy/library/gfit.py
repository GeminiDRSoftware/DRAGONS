
import numpy as np
from scipy.optimize import leastsq
from scipy.interpolate import splev, LSQUnivariateSpline, LSQBivariateSpline

class Gfit(object):
    """Least Square fit of a set of (x,y) points using
     Polynomial, Cubic Splines, Legendre or Chebeyshev polynomial
      
     gfit (x,y,fitname='polynomial',order=3, xlim=None, ylim=None,p0=None)

     Instantiate the class with two arrays of points: x and
     y = f(x).

     Inputs:
     x                 : array of x values
     y                 : array of y values (y = f(x))
     fitname          : 'polynomial' (default), 'legendre', 'chebyshev'
                         or 'cubic' for a cubic spline aproximation
     xlim              : Tuple with (xmin,xmax), default is x[0],x[last]
     ylim              : Tuple with (ymin,ymax), default is y[0],y[last]
     p0                : Initial guess for the coefficients. Default
                         is (1)*order list.

     After initialization, the instance can be called with an array of
     values xp, and will return the interpolated values
     yp = f(xp).

     Examples
     --------
     import gfit
     >>> zfit = gfit.Gfit(x,y,'legendre')
     >>> zfit(xnew)    # Where xnew is an array of points to evaluate the
                       # given polynomial.

        Available values after fitting are:

        zfit.coeff   : Polynomial coefficients of the fit.
                       No coefficients are available for 'cubic'
    
    Acknowledge:  Some of this code has been borrowed from: Neil Crighton
    """

    def __init__(self, x, y, fitname='polynomial',order=3,xlim=None,
                 ylim=None,p0=None):

        self.order = order
        if p0:
           self.p0 = p0
           self.order = len(p0)
        else: self.p0 = [1.0]*order

        funs = ['polynomial','legendre','chebyshev','cubic','gauss']
        if fitname not in funs:
            raise Exception('Function: "'+fitname+'" is not in '+str(funs))

        self.fitname = fitname
        # Load the address of each fitname
        runit = {'polynomial': self.fit_polynomial,
                   'legendre': self.fit_legendre,
                  'chebyshev': self.fit_chebyshev,
                      'cubic': self.fit_cubic,
                      'gauss': self.fit_gauss}
        self.fitfunc = runit[fitname]

        x = np.asfarray(x)
        y = np.asfarray(y)
        if x.size > 0:
            # check all x values are unique
            if len(x) - len(np.unique(x)) > 0:
                raise Exception('non-unique x values were found!')
            cond = x.argsort()      # sort arrays
            x = x[cond]
            y = y[cond]

            if fitname != 'polynomial':
                keep = np.ones(len(x), bool)
                if xlim is not None:
                    keep &= (xlim[0] < x) & (x < xlim[1])
                else:
                    xlim = (x[0],x[-1])
                if ylim is not None:
                    keep &= (ylim[0] < y) & (y < ylim[1])
                x,y = x[keep], y[keep]

                self.ylim = ylim
                self.xlim = xlim
                # This function scaled data to the range [-1,1]
                self.fx = lambda x: 2.*(x - xlim[0])/(xlim[1]-xlim[0]) - 1.
            else:
                self.fx = lambda x: x

            self.x = x
            self.y = y
            self.npts = len(x)

            # Execute the chosen function
            self.fitfunc()
        else:
            self.coeff = [[]]*(order+1)
            self.fx = lambda x: x
            self.evfunc = np.polyval

    def __call__(self,xp):
        """ Given an array of x values, returns 
        interpolated values yp = f(xp) with the function
        self.evfunc(p, xs).
        """

        # make xp into an array
        if not hasattr(xp,'__len__'):  xp = (xp,)
        xp = np.asarray(xp)

        p = self.coeff

        xs = self.fx(xp)    # Scale data if necessary

        yp = self.evfunc(p, xs)

        return yp

    def fit_legendre(self):
        """
          import scipy.special as sp
          L_0 = 1
          L_1 = x
          L_2 = 0.5*(3*x**2 - 1)
          L_3 = 0.5*(5*x**3 - 3*x)           # sp.eval_legendre(3,x)
          L_4 = 0.125*(35*x**4-30*x**2+3)    # sp.eval_legendre(4,x)
  
          Fit (x,y) data using a Legendre polynomial of the form

          y(x) = A + B*L_1 + C*L_2 + D*L_3 + E*L_4 + ...

          using the scipy.optimize.leastsq function. 
        """

        terms = ('p[0]', 'p[1]*x', 'p[2]*0.5*(3*x**2-1)',
                 'p[3]*0.5*(5*x**3-3*x)', 'p[4]*0.125*(35*x**4-30*x**2+3)')
        func = 'lambda p,x:'
        for i in range(self.order):
            func = func+'+'+terms[i] 
        self.evfunc = eval(func)

        e = lambda p,y,x: y - self.evfunc(p,x)

        x = self.x
        y = self.y

        p0 = [1.]*self.order
        xs = self.fx(x)
        coeff,ier = leastsq(e, p0, args=(y, xs))
        self.coeff = coeff
        self.ier = ier
        self.__doc__ ='Legendre function evaluator'

        return

    def fit_chebyshev(self):
        """
          # Chebyshev polynomials of the first kind.
          T_0 = 1
          T_1 = x
          T_2 = 2x**2 - 1
          T_3 = 4x**3 - 3x
          T_4 = 8x**4 - 8x**2 + 1
          T_5 = 16x**5 - 20x**3 + 5x
          T_6 = 32x**6 - 48x**4 + 18x**2 - 1.
  
          Fit (x,y) data using a Legendre polynomial of the form

          y(x) = A + B*T_1 + C*T_2 + D*T_3 + E*T_4 + ...

          using the scipy.optimize.leastsq function. 
        """

        # The p[1]*x term is executed in the function resid.

        terms = ['p[0]','p[1]*x', 'p[2]*(2.*x**2.-1)', 
                 'p[3]*(4.*x**3 - 3.*x)','p[4]*(8*x**4-8*x**2+1)']

        func = 'lambda p,x:'
        for i in range(self.order):
            func = func+'+'+terms[i]
        self.evfunc = eval(func)

        e = lambda p,y,x: y - self.evfunc(p,x)

        x = self.x
        y = self.y
        xs = self.fx(x)

        p0 = [1.]*self.order
        plsq,ier = leastsq(e, p0, args=(y, xs))
        self.coeff = plsq
        self.ier = ier

        return

    def fit_polynomial(self):
        """ Fit (x,y) arrays using a polynomial of the form:
          y(x) = A + Bx + Cx**2 + Dx**3

          by the leastsq methods using scipy.optimize.leastsq  function.

        """

        # Using numpy's seems simpler.
        self.coeff = np.polyfit(self.x, self.y, self.order)
        #self.fx = lambda x: x
        self.evfunc = np.polyval
        
        return 

        terms = ['p[0]','p[1]*x', 'p[2]*x**2', 'p[3]*x**3','p[4]*x**4']
        func = 'lambda p,x:'
        for i in range(self.order):
            func = func+'+'+terms[i] 
        self.func = eval(func)

        self.fx = lambda x: x

        e = lambda p,y,x: y - self.func(p,x)

        x = self.x
        y = self.y

        p0 = [1.]*self.order     # The initial guess
        
        plsq,ier = leastsq(e, p0, args=(y, x))
        self.coeff = plsq
        self.ier = ier

        return

    def fit_cubic(self):
        """Fit (x,y) arrays using a cubic Spline approximation.
           It uses LSQUnivariateSpline from scipy.interpolate
           which uses order-1 knots.
           

        """

        x = self.x
        y = self.y
        xmin,xmax = x.min(),x.max()
        npieces = self.order

        rn = (xmax - xmin)/npieces 
        tn = [(xmin+rn*k) for k in range(1,npieces)]
        zu = LSQUnivariateSpline(x,y,tn)
        
        self.coeff = zu.get_coeffs()
        self.fx = lambda x: x
        self.evfunc = lambda p,x: zu(x)
        self.ier = None
       
        return

    def fit_gauss(self):

        #model = lambda p,t: p[0]+p[1]*np.exp(-((t-p[2])/p[3])**2 )
        model = lambda p,t: p[0]*np.exp(-((t-p[1])/p[2])**2 )
        self.evfunc = lambda p,x: model(p,x)
        e = lambda p,y,x: y - model(p,x)

        x = self.x
        y = self.y
        self.fx = lambda x: x

        # Initial guess
        #p0 = [height,center,width]
        #p0 = [4503.0732, 2026, 19.0]
        
        plsq,ier = leastsq(e, self.p0, args=(y, x))
        self.coeff = plsq
        self.ier = ier


class Eval(Gfit):
    def __init__(self,coeff,fitname='polynomial',order=3,xlim=None):
        evfname = {'polynomial': self.ev_polynomial,
                   'legendre': self.ev_legendre,
                  'chebyshev': self.ev_chebyshev,
                      'cubic': self.ev_cubic,
                      'gauss': self.ev_gauss}
        if fitname=='polynomial': xlim=None
        Gfit.__init__(self,[],[])
        self.evfunc = evfname[fitname]
        self.coeff = coeff
        self.order = order
        self.xlim = xlim
        if xlim != None:
           self.fx = lambda x: 2.*(x - xlim[0])/(xlim[1]-xlim[0]) - 1.

        # Execute the chosen function
        self.evfunc()

    def ev_chebyshev(self):
        terms = ['p[0]','p[1]*x', 'p[2]*(2.*x**2.-1)', 
                 'p[3]*(4.*x**3 - 3.*x)','p[4]*(8*x**4-8*x**2+1)']
        p = self.coeff
        func = 'lambda p,x:'
        for i in range(self.order):
            func = func+'+'+terms[i]
        self.evfunc = eval(func)
    
    def ev_polynomial(self):
        self.evfunc = np.polyval

    def ev_legendre(self):
        terms = ('p[0]', 'p[1]*x', 'p[2]*0.5*(3*x**2-1)',
                 'p[3]*0.5*(5*x**3-3*x)', 'p[4]*0.125*(35*x**4-30*x**2+3)')
        p = self.coeff
        func = 'lambda p,x:'
        for i in range(self.order):
            func = func+'+'+terms[i] 
        self.evfunc = eval(func)

    def ev_cubic(self):
        """
          Given the knots and coefficients of a B-spline representation, 
          evaluate the value of the smoothing polynomial. This is a
          wrapper around the FORTRAN routines splev of FITPACK.

          Inputs:
              xlim: Minimum and maximum values from the original array 
                    that originated the coefficients.calculated the 
              order: Degree of the smoothing spline. Must be <= 5.
              coeff: Array of coefficients calculated from 
                     scipy.interpolate.LSQUnivariateSpline 

        z._eval_args
        Out[478]: 
        IS:: (t,c,k)
        (array([   14.43977485,    14.43977485,    14.43977485,    14.43977485,
                1508.1111519 ,  3001.78252894,  4495.45390599,  5989.12528304,
                5989.12528304,  5989.12528304,  5989.12528304]),
         array([ 6431.54110099,  6197.56743506,  5731.52447642,  5038.62586329,
                4353.0661675 ,  3902.04649884,  3678.22600129,     0.        ,
                   0.        ,     0.        ,     0.        ]),
         3)
        p = self.coeff

        xs = self.fx(xp)    # Scale data if necessary

        yp = self.evfunc(p, xs)

        """
        xmin,xmax = self.xlim
        coeff = self.coeff
        npieces = self.order
        k = self.order
        kc = 3

        rn = (xmax - xmin)/npieces 
        tn = [(xmin+rn*j) for j in range(1,npieces)]
        tn = np.concatenate(([xmin]*(kc+1),tn,[xmax]*(kc+1)))
        coeff = np.concatenate((coeff,[0.]*(kc+1)))
        p = (tn,coeff,kc)
        print tn
        print coeff
        print kc
        func = 'lambda p,x: splev(x,p)'
        self.fx = lambda x: x
        self.evfunc = eval(func) 
        # Redefine coeff  to use the parent __call__
        self.coeff = p

    def ev_gauss(self):
        pass

class InterpCubicSpline:
    """Interpolate a cubic spline through a set of points.

    Instantiate the class with two arrays of points: x and
    y = f(x).

    Inputs:

    x                 : array of x values
    y                 : array of y values (y = f(x))
    firstderiv = None : derivative of f(x) at x[0]
    lastderiv  = None : derivative of f(x) at x[-1]
    nochecks = False  : if False, check the x array is sorted and
                        unique. Set to True for increased speed.

    After initialisation, the instance can be called with an array of
    values xp, and will return the cubic-spline interpolated values
    yp = f(xp).

    The spline can be reset to use a new first and last derivative
    while still using the same initial points by calling the set_d2()
    method.

    If you want to calculate a new spline using a different set of x
    and y values, you'll have to instantiate a new class.

    The spline generation is based on the NR algorithms (but not their
    routines.)

    Examples
    --------

    """
    def __init__(self, x, y, firstderiv=None, lastderiv=None, nochecks=False):
        x = np.asarray(x)
        y = np.asarray(y)
        if 'i' in  (x.dtype.str[1], y.dtype.str[1]):
            raise TypeError('Input arrays must not be integer')
        if not nochecks:
            # check all x values are unique
            if len(x) - len(np.unique(x)) > 0:
                raise Exception('non-unique x values were found!')
            cond = x.argsort()                    # sort arrays
            x = x[cond]
            y = y[cond]

        self.x = x
        self.y = y
        self.npts = len(x)
        self.set_d2(firstderiv, lastderiv)

    def __call__(self,xp):
        """ Given an array of x values, returns cubic-spline
        interpolated values yp = f(xp) using the derivatives
        calculated in set_d2().
        """
        x = self.x;  y = self.y;  npts = self.npts;  d2 = self.d2

        # make xp into an array
        if not hasattr(xp,'__len__'):  xp = (xp,)
        xp = np.asarray(xp)

        # for each xp value, find the closest x value above and below
        i2 = np.searchsorted(x,xp)

        # account for xp values outside x range
        i2 = np.where(i2 == npts, npts-1, i2)
        i2 = np.where(i2 == 0, 1, i2)
        i1 = i2 - 1

        h = x[i2] - x[i1]
        a = (x[i2] - xp) / h
        b = (xp - x[i1]) / h
        temp = (a**3 - a)*d2[i1] +  (b**3 - b)*d2[i2]
        yp = a * y[i1] + b * y[i2] + temp * h**2 / 6.

        return yp

    def _tridiag(self,temp,d2):
        x, y, npts = self.x, self.y, self.npts
        for i in range(1, npts-1):
            ratio = (x[i]-x[i-1]) / (x[i+1]-x[i-1])
            denom = ratio * d2[i-1] + 2.       # 2 if x vals equally spaced
            d2[i] = (ratio - 1.) / denom       # -0.5 if x vals equally spaced
            temp[i] = (y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1])
            temp[i] = (6.*temp[i]/(x[i+1]-x[i-1]) - ratio * temp[i-1]) / denom
        return temp

    def set_d2(self, firstderiv=None, lastderiv=None, verbose=False):
        """ Calculates the second derivative of a cubic spline
        function y = f(x) for each value in array x. This is called by
        __init__() when a new class instance is created.

        optional inputs:

        firstderiv = None : 1st derivative of f(x) at x[0].  If None,
                             then 2nd derivative is set to 0 ('natural').
        lastderiv  = None : 1st derivative of f(x) at x[-1].  If None,
                             then 2nd derivative is set to 0 ('natural').
        """
        if verbose:  print 'first deriv,last deriv',firstderiv,lastderiv
        x, y, npts = self.x, self.y, self.npts
        d2 = np.empty(npts)
        temp = np.empty(npts-1)

        if firstderiv is None:
            if verbose:  print "Lower boundary condition set to 'natural'"
            d2[0] = 0.
            temp[0] = 0.
        else:
            d2[0] = -0.5
            temp[0] = 3./(x[1]-x[0]) * ((y[1]-y[0])/(x[1]-x[0]) - firstderiv)

        temp = self._tridiag(temp,d2)

        if lastderiv is None:
            if verbose:  print "Upper boundary condition set to 'natural'"
            qn = 0.
            un = 0.
        else:
            qn = 0.5
            un = 3./(x[-1]-x[-2]) * (lastderiv - (y[-1]-y[-2])/(x[-1]-x[-2]))

        d2[-1] = (un - qn*temp[-1]) / (qn*d2[-2] + 1.)
        for i in reversed(range(npts-1)):
            d2[i] = d2[i] * d2[i+1] + temp[i]

        self.d2 = d2

def fitrej(x, y, order=3, clip=6, xlim=None, ylim=None,
             mask=None, debug=False):
    """ Fit a legendre polynomial to data, rejecting outliers.

    Fits a polynomial f(x) to data, x,y.  Finds standard deviation of
    y - f(x) and removes points that differ from f(x) by more than
    clip*stddev, then refits.  This repeats until no points are
    removed.

    Inputs
    ------
    x,y:
        Data points to be fitted.  They must have the same length.
    order: int (3)
        Order of polynomial to be fitted.
    clip: float (6)
        After each iteration data further than this many standard
        deviations away from the fit will be discarded.
    xlim: tuple of maximum and minimum x values, optional
        Data outside these x limits will not be used in the fit.
    ylim: tuple of maximum and minimum y values, optional
        As for xlim, but for y data.
    mask: sequence of pairs, optional
        A list of minimum and maximum x values (e.g. [(3, 4), (8, 9)])
        giving regions to be excluded from the fit.
    debug: boolean, default False
        If True, plots the fit at each iteration in matplotlib.

    Returns
    -------
    coeff, x, y:
        x, y are the data points contributing to the final fit. coeff
        gives the coefficients of the final polynomial fit (use
        np.polyval(coeff,x)).

    Examples
    --------
    >>> x = np.linspace(0,4)
    >>> np.random.seed(13)
    >>> y = x**2 + np.random.randn(50)
    >>> coeff, x1, y1 = polyfitr(x, y)
    >>> np.allclose(coeff, [1.05228393, -0.31855442, 0.4957111])
    True
    >>> coeff, x1, y1 = polyfitr(x, y, order=1, xlim=(0.5,3.5), ylim=(1,10))
    >>> np.allclose(coeff, [3.23959627, -1.81635911])
    True
    >>> coeff, x1, y1 = polyfitr(x, y, mask=[(1, 2), (3, 3.5)])
    >>> np.allclose(coeff, [1.08044631, -0.37032771, 0.42847982])
    True
    """

    good = ~np.isnan(x) & ~np.isnan(y)
    x = np.asanyarray(x[good])
    y = np.asanyarray(y[good])
    isort = x.argsort()
    x, y = x[isort], y[isort]

    keep = np.ones(len(x), bool)
    if xlim is not None:
        keep &= (xlim[0] < x) & (x < xlim[1])
    if ylim is not None:
        keep &= (ylim[0] < y) & (y < ylim[1])
    if mask is not None:
        badpts = np.zeros(len(x), bool)
        for x0,x1 in mask:
            badpts |=  (x0 < x) & (x < x1)
        keep &= ~badpts

    x,y = x[keep], y[keep]
    if debug:
        fig = pp.figure()
        ax = fig.add_subplot(111)
        ax.plot(x,y,'.')
        ax.set_autoscale_on(0)
        #pp.show()

    # Fit a legendre Function
    zfit = Gfit(x,y, 'legendre',xlim=xlim)
    fx = zfit.fx
    if debug:
        pts, = ax.plot(x, y, '.')
        #poly, = ax.plot(x, fp(coeff,fx(x)), lw=2)
        poly, = ax.plot(x, zfit(x), lw=2)
        #pp.show()
        raw_input('Enter to continue')
    norm = np.abs(y - zfit(x))
    #norm = np.abs(y - fp(coeff,fx(x)))
    stdev = np.std(norm)
    condition =  norm < clip * stdev
    oxlen= len(x)
    y = y[condition]
    x = x[condition]
    print 'x len bef after,',oxlen,len(x)
    print 'norm, clip',norm.max(),'>',clip*stdev,clip,stdev, zfit.coeff
    while norm.max() > clip * stdev:
        if len(y) < order + 1:
            raise Exception('Too few points left to fit!')
        zfit = Gfit(x,y, 'legendre',xlim=xlim)
        fx = zfit.fx
        if debug:
            pts.set_data(x, y)
            poly.set_data(x, zfit(x))
            #poly.set_data(x, fp(coeff,fx(x)))
            #pp.show()
            raw_input('Enter to continue')
        #norm = np.abs(y - fp(coeff,fx(x)))
        norm = np.abs(y - zfit(x))
        stdev = norm.std()
        condition =  norm < clip * stdev
        oxlen= len(x)
        y = y[condition]
        x = x[condition]
        #print 'x len b a:',oxlen,len(x),stdev, zfit.coeff

    return zfit,x,y
    #return coeff,ier

class Pix2coord():
    def __init__(self,ad):
        """Load parameters from a FITS header into the fitting function such that
           we can evaluate the function.
           The header has keywords of the form COEFF_A, COEFF_B..., from low
           to high order.
           :param xmin,xmax:  Min,max range where to eveluate
           :param fitname:  Fitting function name used.
           :type fitname: {'polynomial','legendre',or 'chebyshev'}
           :param order: polynomial order used to evaluate the coeffs
           :param coeff: Coefficients from the fitting
           :type coeff: List
           :return: An array of evaluated values

           Example:
             ef = gfit.Pix2coord(ad)
             ef(1300)   # Evaluate wavelength at pixel 1300 in the middle row
             ef(1300,400)   # Evaluate wavelength at pixel 1300 at row 400
        """

        tbh = ad['WAVECAL',1].header
        pixsample = np.asfarray(tbh['pixrange'].split())
        self.fitname = tbh['fitname']
        self.order = tbh['fitorder']
        self.coeff = np.asfarray(tbh['fitcoeff'].split())
        # Set up the pix2wavelength evaluator function 
        xmin, xmax = pixsample
        self.xnorm = lambda x: 2.*(x - xmin) / (xmax-xmin) - 1

        f3dcoeff = []
        for k in ['A', 'B', 'C', 'D', 'E', 'F']:
            f3dcoeff.append(tbh['COEFF_'+k])
        self.f3dcoeff = np.asfarray(f3dcoeff)

     
    def __call__(self,x,y=None):
        if not hasattr(x,'__len__'):  x = (x,)
        x = np.asarray(x)
        if y == None:
           return self.fx(x)
        else:
           if not hasattr(y,'__len__'):  y = (y,)
           y = np.asarray(y)
           return self.fx(x-self.fxy(x,y))

    def fxy(self, x,y):
        """ Sets up the 3d fit evaluation function
        """    
        coeff = self.f3dcoeff 
        fxy =  coeff[0] + coeff[1]*x + coeff[2]*y + \
                    coeff[3]*x*x + coeff[4]*y*y + coeff[5]*x*y
        return fxy

    def fx(self,x):

        terms = {
        'polynomial': ['p[0]','p[1]*xn', 'p[2]*xn**2', 'p[3]*xn**3','p[4]*xn**4'],
        'legendre'  : ['p[0]', 'p[1]*xn', 'p[2]*0.5*(3*xn**2-1)',\
                       'p[3]*0.5*(5*xn**3-3*xn)', 'p[4]*0.125*(35*xn**4-30*xn**2+3)'],
        'chebyshev' : ['p[0]','p[1]*xn', 'p[2]*(2.*xn**2.-1)',\
                       'p[3]*(4.*xn**3 - 3.*xn)','p[4]*(8*xn**4-8*xn**2+1)'],
                }
        # cubic: NYI

        tpol = terms[self.fitname]
        p = self.coeff
        xn = self.xnorm(x)
        func = 'lambda p,xn:'
        for i in range(self.order):
            func = func+ '+' +tpol[i] 
        ff =  eval(func)
        return ff(p,xn)


class Fit3d():

    def __init__(self,x,y,z,function='polynomial',order=4,xlim=None,ylim=None):
        """Fit a surface such that z = f(x,y) using a regular, Legendre
           or Chebyshev polynomial.

           Input
           -----
             function: 'polynomial' is the default value. Others are
                       'legendre' or 'chebyshev'.

             order: 4 is the default value. Orders can be 2 for 4 terms, 3
                    for 9 terms or 4 for 16 terms polynomials. 

           Note: For polynomial fitting the function is fixed at 6 terms.


           Examples
           --------

           Assuming you have sparse data points (x_array,y_array) with
           values z_array at each of these points. We want to fit these
           points with a legendre polynomial of order 4.
           
           import gfit 
           import numpy as np

           zfit = gfit.Fit3d(x_array,y_array,z_array,'legendre',4)
           
           Now evaluate the function at some area between the boundaries
           of (x_array,y_array).

           x,y = np.mgrid[0:5,0:5]
           
           print zfit(x,y) 
           
         
        """
        fitname = ['polynomial','legendre','chebyshev','cubic']
        if function not in fitname:
            raise Exception('Function: "'+function+'" is not in '+str(fitname))

        # Load the address of each function.
        runit = {'polynomial': self.fit3d_polynomial,
                   'legendre': self.fit3d_legendre,
                  'chebyshev': self.fit3d_chebyshev,
                  'cubic':     self.fit3d_cubic,
                }
        self.fitfunc = runit[function]
        self.fitname = function
     
        if order not in [2,3,4] and (function!='polynomial'):
            raise Exception('Order needs to be 2, 3 or 4. for Chebyshev'
                ' or Legendre.')

        self.order = order

        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        if len(x.shape) > 1:
           x = x.flatten()
           y = y.flatten()
           z = z.flatten()

        cond = x.argsort()      # sort arrays
        x = x[cond]
        y = y[cond]
        z = z[cond]

        if xlim == None:
           xlim = (x.min(),x.max())
        if ylim == None:
           ylim = (y.min(),y.max())
        self.xlim= xlim
        self.ylim= ylim
        self.deltx = xlim[1]-xlim[0]
        self.delty = ylim[1]-ylim[0]

        # This function scaled data to the range [-1,1]
        self.fx = lambda x: 2.*(x - xlim[0])/(xlim[1]-xlim[0]) - 1.
        self.fy = lambda y: 2.*(y - ylim[0])/(ylim[1]-ylim[0]) - 1.

        self.x = x
        self.y = y
        self.z = z
        self.xs = self.fx(x)    # Scaled data
        self.ys = self.fy(y)    # Scaled data
        self.npts = len(x)

        # Execute the chosen function
        self.fitfunc()

        return
 
    def poly_terms(self):
        # These are the indices for polynomial of order 2x2, 3x3 and 4x4.
        terms_indx = {2: [0,1,4,5],3: [0,1,2,4,5,6,8,9,10],4:range(16)}

        # These are the 16 terms to build a 4x4 polynomial.
        terms = \
          ['1   ', 'x     ', 'a(x)     ', 'b(x)     ', \
           'y   ', 'x*y   ', 'a(x)*y   ', 'b(x)*y   ', \
           'a(y)', 'x*a(y)', 'a(x)*a(y)', 'b(x)*a(y)', \
           'b(y)', 'x*b(y)', 'a(x)*b(y)', 'b(x)*b(y)'
          ]
        order = min(self.order,4)
        func = 'lambda coeff,x,y:'
        k = 0
        for i in terms_indx[order]:
            func = func + '+coeff[%d]*%s'%(k,terms[i])
            k+=1

        return func    

    def fit_leastsq(self):
        
        polyn = self.poly_terms()
        self.func = eval(polyn,{'a':self.a,'b':self.b})

        f = lambda p,z,x,y:(self.func(p,x,y)-z)

        #order = 6
        p0 = [1.]*self.order*self.order
        p0 = np.asarray(p0)

        xs = self.xs
        ys = self.ys
        z  = self.z
        coeff,ier = leastsq(f,p0,args=(z,xs,ys),full_output=False)

        self.coeff = coeff
        self.ier = ier


    def fit3d_polynomial(self):
        """
          Polynomial surface fitting function:
          F(x,y) = c[0] + c[1]*x + c[2]*y + c[3]*x*x + c[4]*y*y + c[5]*x*y

          NOTE: The order of the polynomial is set to 6.

        """
        from scipy.optimize import leastsq

        self.a = lambda v: v**2
        self.b = lambda v: v**3

        self.fit_leastsq()

        return 
    
    def fit3d_chebyshev(self):
        """
          Fit a Chebyshev polynomial to a set of (x_array,y_array) points
          using the LSQ method. For each point (x,y) there is a value z. 

          Least squares fit using the output variable as the dependent variable.

          fit(x,y) = SUM[m,n](Cmn * Pmn)    for m,n=0,1,2,3
    
          where the Cmn are the coefficients of the polynomial terms and Pmn
          is:

             Pmn = Pm*Pn                 for m,n=0,1,2,3

          Then for an order 2 we have:
          F(x,y) = C00*P00 + C10*P10 + C20*P20 +
                   C01*P01 + C11*P11 + C21*P21 + C22*P22 
          P0(v) = 1
          P1(v) = v
          P(n+1)(v) = (2*v*Pn(v) - P(n-1)(v)
        
          P00 = P0(x)*P0(y)
          P10 = P1(x)*P0(y)
          P20 = P2(x)*P0(y)
          P20 = 2*x*x - 1

          
          T_0 = 1
          T_1 = x
          T_2 = 2x**2 - 1
          T_3 = 4x**3 - 3x
          T_4 = 8x**4 - 8x**2 + 1
          T_5 = 16x**5 - 20x**3 + 5x
          T_6 = 32x**6 - 48x**4 + 18x**2 - 1.

          A Chebyshev polynomial of order 4 is:

          F(x,y) = 
            p[0]       + p[1]*x       + p[2]*a(x)       + p[3]*b(x)       + 
            p[4]*y     + p[5]*x*y     + p[6]*a(x)*y     + p[7]*b(x)*y     + 
            p[8]*a(y)  + p[9]*x*a(y)  + p[10]*a(x)*a(y) + p[11]*b(x)*a(y) + 
            p[12]*b(y) + p[13]*x*b(y) + p[14]*a(x)*b(y) + p[15]*b(x)*b(y)

          where a and b are:
            a(v) = (2*v**2 - 1)
            b(v) = (4*v**3 - 3*v)
      
        """
        from scipy.optimize import leastsq

        self.a = lambda v: (2*v**2 - 1)      
        self.b = lambda v: (4*v**3 - 3*v)   

        self.fit_leastsq()

        return 
            
    def fit3d_legendre(self):
        """
          Fit a Legendre polynomial to a set of (x_array,y_array) points
          using the LSQ method. For each point (x,y) there is a value z. 

          Input: 
              (x_array,y_array,z_array).
                 The input arrays.  Where each coordinate (x,y)
                 have a value z.
              Order. 
                 Order of the polynomial to fit. The supported orders
                 are 2,3,or 4. See below for examples.
                
          The polynomial has the form:
          fit(x,y) = SUM[m,n](Cmn * Pmn)    for m,n=0,1,2,3
    
          where the Cmn are the coefficients of  the  polynomials  terms  Pmn,
          and the Pmn are defined as follows:   (for m,n: 0,1,2)
          fit(x,y) = C00*P00 + C10*P10 + C20*P20 +
                             C01*P01 + C11*P11 + C21*P21 + C22*P22 
          P0(v) = 1
          P1(v) = v
          P(n+1)(v) = ((2n+1)*v*Pn - n*P(n-1)) /(n+1)
          P2(v) = (3*v*P1 - 1)/2
          P2(v) = (3*v*v - 1)/2
        
          P00 = P0(x)*P0(y)
          P10 = P1(x)*P0(y)
          P20 = P2(x)*P0(y)
          P20 = (3*x**2 - 1)/2

          # The Legendre polynomial of degree 4 to fit is:
          F(x,y)=
            C[0]       + C[1]*x       + C[2]*a(x)       + C[3]*b(x)       + 
            C[4]*y     + C[5]*x*y     + C[6]*a(x)*y     + C[7]*b(x)*y     + 
            C[8]*a(y)  + C[9]*x*a(y)  + C[10]*a(x)*a(y) + C[11]*b(x)*a(y) + 
            C[12]*b(y) + C[13]*x*b(y) + C[14]*a(x)*b(y) + C[15]*b(x)*b(y)
          
          # The Legendre polynomial of degree 3 to fit is:
          F(x,y)=
            C[0]       + C[1]*x       + C[2]*a(x)     
            C[3]*y     + C[4]*x*y     + C[5]*a(x)*y    
            C[5]*a(y)  + C[7]*x*a(y)  + C[8]*a(x)*a(y)
          
          # The Legendre polynomial of degree 2 to fit is:
          F(x,y)=
            C[0]       + C[1]*x      
            C[2]*y     + C[5]*x*y   
          
        """
        from scipy.optimize import leastsq

        self.a = lambda v: 0.5*(3*v**2-1)
        self.b = lambda v: 0.5*(5*v**3-3*v)

        self.fit_leastsq()

        return 
    
    def fit3d_cubic(self):
        # Create the knots (10 knots in each direction)
        x = self.x.ravel()
        y = self.y.ravel()
        #x = self.xs
        #y = self.ys
        z = self.z.ravel()
        xkts = np.linspace(5, x.max()-5, 0)
        ykts = np.linspace(5, y.max()-5, 0)

        zspl = LSQBivariateSpline(x,y,z,xkts,ykts,eps=None)

        self.coeff = zspl.get_coeffs()
        self.fx = lambda x: np.asarray(x).ravel()
        self.fy = lambda y: np.asarray(y).ravel()
        self.evfunc = zspl
        self.ier = None

    def _fit3d_eval_cubic(self,x,y):
        """
          Evaluate the cubic spline function using
          the method LSQBivariateSpline.ev() which
          needs to same number of elements in x and y.

          fit3d_eval_cubic supports (indx, y_array) or
          (x_array,indx), where indx is one element.
          e.g. (12,x_array)

        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        #x = self.fx(x).ravel()
        #y = self.fy(y).ravel()
        if (x.size == 1) and (y.size != 1):
           x = x.repeat(y.size)
        elif (y.size == 1) and (x.size != 1):
           y = y.repeat(x.size)
        
        return self.evfunc.ev(x,y)
        
        
    def __call__(self,x,y):
     
        if self.fitname=='cubic':
            yp = self._fit3d_eval_cubic(x,y)
            return yp
          
        x = np.asarray(x)
        y = np.asarray(y)
        x = x.flatten()
        y = y.flatten()
        p = self.coeff

        x = self.fx(x)    # Scaled data
        y = self.fy(y)    

        yp = self.func(p, x,y)
        return yp

class Eval2(object):
    def __init__(self,fitname,order,coeff,xlim,ylim):
        functions = ['polynomial','legendre','chebyshev','cubic']
        if  fitname not in functions:
            raise Exception('Function: "'+fitname+'" is not in '+str(functions))

        # Load the address of each function.
        runit = {'polynomial': self.eval_poly2,
                   'legendre': self.eval_legendre2,
                  'chebyshev': self.eval_chebyshev2,
                  'cubic':     self.eval_cubic2,
                }
        self.evfunc = runit[fitname]
        self.fitname = fitname

        if order not in [2,3,4] and (function!='polynomial'):
            raise Exception('Order needs to be 2, 3 or 4. for Chebyshev'
                ' or Legendre.')

        self.order = order
        self.coeff = coeff

        self.fx = lambda x: 2.*(x - xlim[0])/(xlim[1]-xlim[0]) - 1.
        self.fy = lambda y: 2.*(y - ylim[0])/(ylim[1]-ylim[0]) - 1.
        
    def __call__(self,x,y):

        if self.fitname=='cubic':
            yp = self._fit3d_eval_cubic(x,y)
            return yp

        x = np.asarray(x)
        y = np.asarray(y)
        x = x.flatten()
        y = y.flatten()
        p = self.coeff

        x = self.fx(x)    # Scaled data
        y = self.fy(y)
        return self.evfunc(x,y)

    def poly_terms(self):
        # These are the indices for polynomial of order 2x2, 3x3 and 4x4.
        terms_indx = {2: [0,1,4,5],3: [0,1,2,4,5,6,8,9,10],4:range(16)}

        # These are the 16 terms to build a 4x4 polynomial.
        terms = \
          ['1   ', 'x     ', 'a(x)     ', 'b(x)     ', \
           'y   ', 'x*y   ', 'a(x)*y   ', 'b(x)*y   ', \
           'a(y)', 'x*a(y)', 'a(x)*a(y)', 'b(x)*a(y)', \
           'b(y)', 'x*b(y)', 'a(x)*b(y)', 'b(x)*b(y)'
          ]
        order = min(self.order,4)
        func = 'lambda x,y:'
        k = 0
        for i in terms_indx[order]:
            func = func + '+coeff[%d]*%s'%(k,terms[i])
            k+=1

        return func    

    def eval_poly2(self,x,y):
        a = lambda v: v**2
        b = lambda v: v**3

        terms = self.poly_terms()
        fu = eval(terms,{'coeff':self.coeff,'a':a,'b':b})
        return fu(x,y)
        
    def eval_legendre2(self,x,y):
        a = lambda v: 0.5*(3*v**2-1)
        b = lambda v: 0.5*(5*v**3-3*v)

        terms = self.poly_terms()
        fu = eval(terms,{'coeff':self.coeff,'a':a,'b':b})
        return fu(x,y)
        

    def eval_chebyshev2(self,x,y):
        a = lambda v: 2.0*v**2-1.0
        b = lambda v: 4.0*v**3-3.0*v

        terms = self.poly_terms()
        fu = eval(terms,{'coeff':self.coeff,'a':a,'b':b})
        return fu(x,y)
        

    def eval_cubic2(self,x,y):
        pass

def TTread_wavecal(ad):

    #read WAVECAL extension

    data = ad['WAVECAL'].data

    ev_function={}
    for row in [0,1]:
        tb = data[row] 
        order = tb.field('order')
        coeff = tb.field('coeff')
        xlim = tb.field('xlim')
        ylim = tb.field('ylim')
        fitname = tb.field('fitname')
        mode = tb.field('fitmode')
        ev_function[mode] = Eval2(fitname,order,coeff,xlim,ylim)

    return ev_function


def TTlinearize_image(image,zzfunction):
    """
      Linearize image using the 2D funcion zz


     Linearize the image applying the surface function 
     evaluator from the method 'fit_image'.         

     Using the inverse function evaluator for each row of the 
     input image:
     pixel = self.zinv(lambda, row)

     Generating a set of lambdas with a dispertion value
     (cdelt = (self.z(nx) - self.z(1))/nx) as 
     (lambdas = (ixx-crpix)*cdelt + crval), where 'ixx' is the
     array of indices (1..nx) along the dispersion axis.
     With the inverse function we obtain the pixel coordinates
     corresponding to each lambda value. Interpolating the
     input image values at each of these new pixel coordinates
     using spline interpolation we linearize the input image.
     
     *Input*

        image: (None) 
           If an ndarray is supplied use this one as the input
           image to linearize. It should have the same dispersion
           characteristics since it uses the ARC image dispersion
           and fitting functions.
           
        zzfunction: 
           Dictionary with functions evaluator {'zzfit' and 'zzinvfit'}.
           These are constructed using the class Eval2 (see above)
          
              

     *Output*
     
        new_image: 
           Ndarray of the same shape as the input image.


    """  
    import ginterpolate as gi

    zz = zzfunction['zzfit']
    zinv = zzfunction['zinvfit'] 
    data = image 

    # The output arrays
    ny,nx = np.shape(data)
    outdtype = data.dtype
    linear_im = np.zeros((ny,nx),dtype=outdtype)
    ym = ny/2

    z = lambda x: zz(x,ym)
    # ----- Linearize ------
    
    # Dispersion. We use the already calculated mapping function z
    # such that lambda=z(pixels).  The dispersion direction is 
    # along the x-axis.
    cdelt = (z(nx) - z(1))/nx
    crpix = 1
    if cdelt < 0:
        crval = z(nx)
        cdelt = -cdelt
    else:
        crval = z(1)

    # The linear array in wavelength units for the dispersion axis 
    lams = (np.arange(nx)-crpix)*cdelt + crval
    lamf = lambda x: (x-crpix)*cdelt + crval

    for y in range(ny):
        # Calculate the pixel position for each lambda
        # using the inverse function.
        pixels = zinv(lams,y)
        line = gi.ginterpolate(data[y,:],pixels,order=4)
        linear_im[y,:] = np.asarray(line,dtype=outdtype)    # Keep the input datatype

    # Calculate the linear WCS 
    #self.linear_wcs((ny,nx))

    return linear_im


















