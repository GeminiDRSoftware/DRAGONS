"""
geomapy1beta.py
===============

An Image Coordinate Transformation based on python.
Taking two arrays as input, this program could compute
a function that will be the transformation required to
map one array, taked as a reference coordinate system,
to the second array, taked as the input coordinate system.

The computed transformation is returned as a lambda
function that has the form::

                        # xin = f(xref,yref)
                        # yin = g(xref,yref)
                        ####################

These two functions, f & g, are either a power series
polynomial, a Legendre polynomial or a Chebyshev 
polynomial,  of  order  xxorder and xyorder in x and
yxorder and yyorder in y. The default order for each
of these is 2.

Also, exists the options 'none','half' or 'full' that
are used to complete the polynomial cross terms, being
the default 'half'.

As fitgeometry the user can choose between a 'general',
'shift', 'xyscale', 'rotate', 'rscale', or 'rxyscale',
set as default the option 'general'.

The region of validity of the fit is defined by setting
the values of xmin, xmax, ymin and ymax. These parameters
can be used to reject out of range data before the actual
fitting is done.

The transformation computed by the "general" fittinggeometry
is arbitrary and does not correspond to a physically 
meaningful model. However the computed coefficients for 
the linear term can be given a simple geometrical geometric 
interpretation for all the fitting geometries as shown below.::

                        fitting geometry = general (linear term)
                            xin = a + b * xref + c * yref
                            yin = d + e * xref + f * yref

    
The coefficients can be interpreted as follows. Xref0, yref0,  xin0,
yin0   are   the   origins   in   the  reference  and  input  frames 
respectively. Orientation and skew are the rotation of the x  and  y
axes  and  their  deviation from perpendicularity respectively. Xmag
and ymag are the scaling factors in x and y and are  assumed  to  be
positive.::
    
                        general (linear term)
                            xrotation = rotation - skew / 2
                            yrotation = rotation + skew / 2
                            b = xmag * cos (xrotation)
                            c = ymag * sin (yrotation)
                            e = -xmag * sin (xrotation)
                            f = ymag * cos (yrotation)
                            a = xin0 - b * xref0 - c * yref0 = xshift
                            d = yin0 - e * xref0 - f * yref0 = yshift

*** Part of these documentation can be found in the program of IRAF, U{Geomap<http://iraf.noao.edu/scripts/irafhelp?geomap>}, which is
the program in which is based geomapy1beta ***

@author: U{Juan S. Catalan Olmos<mailto:jscatala@alumnos.inf.utfsm.cl>}
@contact: U{jscatala@alumnos.inf.utfsm.cl<mailto:jscatala@alumnos.inf.utfsm.cl>}
@Organization: Gemini Observatory - Aura Inc. U{http://www.gemini.edu}
@version: 0.1.0 || Feb. , 2009
@status: Under Development.

"""
import sys
import numpy as np
import math as mt
from os import path
from numpy import ndarray, mean, dot
from numpy import linalg as NLA
from scipy import stats

class CoordTransform(object):
    """
    Object that take arrays of matching coodinate pairs
    to inizalize a transform object.
    """

    def __init__(self,xyref=[],xyin=[],dbname='default.db', xmin='INDEF',xmax='INDEF',ymin='INDEF',ymax='INDEF',exparam=''):

        """
        Initialize the general value of the Fit Object.
        @param xyref: Array with the xyref points. Also could be 
           the string "help()", that send the user to the help module.
        @type xyref: Array | String
        @param xyin: Array with the xyin points.
        @type xyin: Array
        @param dbname: The name of the text file database
             where the computed transformations will be stored
        @type dbname: String
        @param xmin,xmax,ymin,ymax: (Default: 'INDEF') The  range  
             of  reference  coordinates  over  which the computed 
             coordinate transformation is valid. If is '' (NULL) or 
             the string 'INDEF', then the minimum and maximum xref 
             and yref values in xyref are used.
        @type xmin: Float
        @type xmax: Float
        @type ymin: Float
        @type ymax: Float
        """

        if not isinstance(xyref,ndarray)and xyref=='help()':
            print 'Help Module. Not implemented YET'
        else:
            #Check if the arrays have the right shape n line , 2 col.
            assert xyref!=[], "Error: No xyref input array given"
            assert xyin!=[], "Error: No xyin input array given"
            assert xyref.shape == xyin.shape, 'ERROR: xyref and xyin have different shape'
            assert xyref.shape[1] == 2, 'ERROR: xyref his not a 2 col. shape array'
            assert xyin.shape[1] == 2, 'ERROR: xyin his not a 2 col. shape array'
            assert len(xyref)==len(xyin), "ERROR: Lenght of input arrays are not equals"

            self.n=len(xyref)
            if not isinstance(xyref,ndarray):
                xyref=np.array(xyref)
            if not isinstance(xyin,ndarray):
                xyin=np.array(xyin)
            self.xyref=[]
            self.xyin=[]
            for i in xrange(self.n):
                self.xyref+=[[xyref[i,0],xyref[i,1]]]
                self.xyin+=[[xyin[i,0],xyin[i,1]]]
            if not isinstance(self.xyref,ndarray):
                self.xyref=np.array(self.xyref,float)
            if not isinstance(self.xyin,ndarray):
                self.xyin=np.array(self.xyin,float)
            if dbname=='':
                    dbname='default.db'
            #if dbname=='default.db':
            #    print 'Warning: No database file specify. Default\
            #           name set (\"default.db\")'
            self.dbname = str(dbname)

            self.xmin = self.numChecker(xmin,1)
            self.xmax = self.numChecker(xmax,1024)
            self.ymin = self.numChecker(ymin,1)
            self.ymax = self.numChecker(ymax,1024)

            #Dict of extra arguments
            self.exargs={'transforms':'' ,'results':'' ,'fitgeometry':'','function':'','xxorder':'','xyorder':'','yxorder':'','yyorder':'','maxiter':'','reject':'','calctype':'','verbose':'','interactive':'','graphics':'','cursor':'','xxterms':'','yxterms':''}
            #End Dict
            self.splitParams(exparam)
            self.paramcalls=[self.xyref,self.xyin,self.dbname,self.xmin,self.xmax,self.ymin,self.ymax,exparam]
            self.fitObj='NONE'
            

    def numChecker(self,num,df):
    #def numChecker(self,num,df,msg):
        """
        Checker for the min and max values. If mun it's not a number, 
           nor INDEF, this function as to the user for a right value.
        @param num: number given by the user.
        @type num: String | Int | Float 
        @param df: Default value (INDEF)
        @type df: String
        @param msg: Message that has to be show in the screen
        @type msg: String
        @return: Float number veryfied its consistency or the 'INDEF' value
        @rtype: Float | String
        """
        if num == '' or num == 'INDEF':
            num = float(df)
        #else:
        #    if num != 'INDEF' :
        #        done = 't'
        #        while done  == 't' :
        #            try:
        #                temp = float (num) +1
        #                done = 'f'
        #            except:
        #                print "Fatal Error: Your input has not been recognised"
        #                num = raw_input(msg)
        #                if num == 'INDEF':
        #                    done = 'f'
        #if num != 'INDEF' :
        #    num = float(num)
        return float(num)
#

    def splitParams(self,exparam):
        """
        Build a dictionary with all the info related with the fit 
           parameters, like fitgeometry and order.
        @param exparam: All the values that won't be set as default.
        @type exparam: String
        """

        if exparam != '' :
            exparam=exparam.split(' ')
            assert len(exparam) < 17 , 'Args Error: Too many arguments'
            for arg in xrange(len(exparam)):
                arg=exparam[arg]
                finder='f'
                for iter in self.exargs.iterkeys():
                    if arg.startswith(iter):
                        assert self.exargs[iter]=='', 'Args Error: \'%s\' \
                             is given more than once' % arg
                        arg=arg.split('=')
                        self.exargs[iter]=arg[1]
                        finder='t'
                        break
                assert finder!='f','Args Error: \'%s\' is not a \
                                valid argument' % arg
        if self.exargs['fitgeometry']== '' :
            self.exargs['fitgeometry']='general'
        else:
            while self.exargs['fitgeometry']!= 'shift' and self.exargs['fitgeometry']!= 'xyscale' and self.exargs['fitgeometry']!= 'rotate' and self.exargs['fitgeometry']!= 'rscale' and self.exargs['fitgeometry']!= 'rxyscale' and self.exargs['fitgeometry']!= 'general' :
                self.exargs['fitgeometry']=raw_input('Fitting geometry (|shift|xyscale|rotate|rscale|rxyscale|general|) (%s): '%self.exargs['fitgeometry'])
        if self.exargs['function']=='':
            self.exargs['function']='polynomial'
        if self.exargs['xxorder']=='':
            self.exargs['xxorder']=2
        if self.exargs['xyorder']=='':
            self.exargs['xyorder']=2
        if int(self.exargs['xxorder']) < 2:
            self.exargs['xxorder']=self.orderChecker('x','x',self.exargs['xxorder'])
        if int(self.exargs['xyorder']) < 2:
            self.exargs['xxorder']=self.orderChecker('x','y',self.exargs['xyorder'])
        if self.exargs['yxorder']=='':
            self.exargs['yxorder']=2
        if self.exargs['yyorder']=='':
            self.exargs['yyorder']=2
        if int(self.exargs['yxorder']) < 2:
            self.exargs['yxorder']=self.orderChecker('y','x',self.exargs['yxorder'])
        if int(self.exargs['yyorder']) < 2:
            self.exargs['yyorder']=self.orderChecker('y','y',self.exargs['yyorder'])
        if self.exargs['xxterms']=='':
            self.exargs['xxterms']='half'
        if self.exargs['yxterms']=='':
            self.exargs['yxterms']='half'
        if self.exargs['maxiter']=='':
            self.exargs['maxiter']=0
        if self.exargs['reject']=='':
            self.exargs['reject']=3.0
        if self.exargs['calctype']=='':
            self.exargs['calctype']='real'
        if self.exargs['verbose']=='':
            self.exargs['verbose']='yes'
        if self.exargs['interactive']=='':
            self.exargs['interactive']='yes'
        if self.exargs['graphics']=='':
            self.exargs['graphics']='stdgraph'
        if self.exargs['transforms']=='':
            self.exargs['transforms']='NULL'
        if self.exargs['results'] != '':
            assert not path.isfile(self.exargs['results']) ,'ERROR: Operation would overwrite existing file (%s)\n'%self.exargs['results']


    def doFit(self):
        """
        Function that calls the specific fit object and after the fit creates the lamba function.
        """
        if self.exargs['function'] == 'polynomial':
            Fit = PolynomialCoordTransform(self.paramcalls)
        elif self.exargs['function'] =='legendre':
            Fit=LegendreCoordTransform(self.paramcalls)
        elif self.exargs['function']=='chebyshev':
            Fit=ChebyshevCoordTransform(self.paramcalls)
        else:
            print 'ValueError: Parameter function: value \'%s\' is not in choice list (chebyshev|legendre|polynomial)\n'%self.exargs['function']
            sys.exit()

        if self.exargs['fitgeometry'] == 'shift':
            Fit.shift()
        elif self.exargs['fitgeometry'] == 'xyscale':
            Fit.xyscale()
        elif self.exargs['fitgeometry'] == 'rotate':
            Fit.rotate()
        elif self.exargs['fitgeometry'] == 'rscale':
            Fit.rscale()
        elif self.exargs['fitgeometry'] == 'rxyscale':
            Fit.rxyscale()
        elif self.exargs['fitgeometry'] == 'general':
            Fit.fxyr()
        else:
            #will never reach this zone, but in case...
            print 'Arg Error: \'%s\' is not a valid fit function' \
                  % self.exargs['fitgeometry']
            sys.exit()

        self.fitObj=Fit
        return Fit.getLambda()

    def info(self):
        """
        Prints in screen all the informations and parameters used to generate the fit object.
        """
        
        keys=['input','database','xmin','xmax','ymin','ymax','transforms','results','fitgeometry','function','xxorder','xyorder','xxterms','yxorder','yyorder','yxterms','maxiter','reject','calctype','verbose','interactive','graphics','cursor']
        msg={'input':'The input coordinate arrays','database':'The output database file','xmin':'Minimum x reference coordinate value','xmax':'Maximum x reference coordinate value','ymin':'Minimum y reference coordinate value','ymax':'Maximum y reference coordinate value','transforms':'The output transform records names','results':'The optional results summary files','fitgeometry':'Fitting geometry','function':'Surface type','xxorder':'Order of x fit in x','xyorder':'Order of x fit in y','xxterms':'X fit cross terms type','yxorder':'Order of y fit in x','yyorder':'Order of y fit in y','yxterms':'Y fit cross terms type','maxiter':'Maximum number of rejection iterations','reject':'Rejection limit in sigma units','calctype':'Computation type','verbose':'Print messages about progress of task ?','interactive':'Fit transformation interactively ?','graphics':'Default graphics device','cursor':'Graphics cursor'}
        print "   %s\t=   xyref  xyin\t\t%s"%(keys[0],msg[keys[0]])
        print "   %s\t=   %s\t\t%s"%(keys[1],self.dbname,msg[keys[1]])
        print "   %s\t\t=   %f\t\t%s"%(keys[2],self.xmin,msg[keys[2]])
        print "   %s\t\t=   %f\t\t%s"%(keys[3],self.xmax,msg[keys[3]])
        print "   %s\t\t=   %f\t\t%s"%(keys[4],self.ymin,msg[keys[4]])
        print "   %s\t\t=   %f\t\t%s"%(keys[5],self.ymax,msg[keys[5]])
        print "   %s\t=   %s\t\t%s"%(keys[6],self.exargs[keys[6]],msg[keys[6]])
        temp=self.exargs[keys[7]]
        if temp== '':
            temp='\'\'\t'
        print "   %s\t=   %s\t\t%s"%(keys[7],temp,msg[keys[7]])
        print "   %s\t=   %s\t\t%s"%(keys[8],self.exargs[keys[8]],msg[keys[8]])
        print "   %s\t=   %s\t\t%s"%(keys[9],self.exargs[keys[9]],msg[keys[9]])
        print "   %s\t=   %i\t\t\t%s"%(keys[10],self.exargs[keys[10]],msg[keys[10]])
        print "   %s\t=   %i\t\t\t%s"%(keys[11],self.exargs[keys[11]],msg[keys[11]])
        print "   %s\t=   %s\t\t%s"%(keys[12],self.exargs[keys[12]],msg[keys[12]])
        print "   %s\t=   %i\t\t\t%s"%(keys[13],self.exargs[keys[13]],msg[keys[13]])
        print "   %s\t=   %i\t\t\t%s"%(keys[14],self.exargs[keys[14]],msg[keys[14]])
        print "   %s\t=   %s\t\t%s"%(keys[15],self.exargs[keys[15]],msg[keys[15]])
        print "   %s\t=   %i\t\t\t%s"%(keys[16],self.exargs[keys[16]],msg[keys[16]])
        print "   %s\t=   %f\t\t%s"%(keys[17],self.exargs[keys[17]],msg[keys[17]])
        print "   %s\t=   %s\t\t%s"%(keys[18],self.exargs[keys[18]],msg[keys[18]])
        print "   %s\t=   %s\t\t\t%s"%(keys[19],self.exargs[keys[19]],msg[keys[19]])
        print "   %s\t=   %s\t\t\t%s"%(keys[20],self.exargs[keys[20]],msg[keys[20]])
        print "   %s\t=   %s\t\t%s"%(keys[21],self.exargs[keys[21]],msg[keys[21]])
        temp=self.exargs[keys[22]]
        if temp== '':
            temp='\'\'\t'
        print "   %s\t=   %s\t\t%s"%(keys[22],temp,msg[keys[22]])
        temp=(raw_input('\n Want to see xyref and xyin? (Be aware that could be long amount of data)[y/n]: '))
        while(temp!='y' and temp!='n'):
            print '%s is not a valid option. Choose between \'y\' or \'n\''%temp
            temp=(raw_input('\n Want to see xyref and xyin? (Be aware that could be long amount of data)[y/n]: '))
        if temp=='y':
            print 'xyref: \n',self.xyref
            print 'xyin: \n',self.xyin


    def writeto(self):
        """
        Function that takes the object self.fitObj and send it to a function or method that cares about this action.

        ****DEVELOPER NOTE: ****
        One possible way to take care of the output is using the module that Phil Hodge wrote to read and write the IRAF database files.        
        """
        assert self.fitObj!='NONE','ERROR: The Fit Object is not initialized yet. In order to use this function call before the function doFit().'
#        Send the object to he handler.
                    

class PolynomialCoordTransform(CoordTransform):
    """
    Object that is created when the fit function is a Power Series Polynomial.
    """

    def __init__(self,params):
        """
        Initialize the Power Series fit object.
        @param params: string with the input arrays, names and parameters to do the fit.
        @type params: Array
        """

        CoordTransform.__init__(self,params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7])
        self.means=np.array([mean(self.xyref[:,0]),mean(self.xyref[:,1]),mean(self.xyin[:,0]),mean(self.xyin[:,1])])
        self.fitObj='SEC'
        self.xshift=0
        self.yshift=0
        self.xmag=0
        self.ymag=0
        self.xrot=0
        self.yrot=0
        self.xyfit=[]
        self.residuals=[]
        self.xrms=0
        self.yrms=0
        self.a=0
        self.b=0
        self.c=0
        self.d=0
        self.e=0
        self.f=0

    def getMag(self,ref,iput):
        """
        Calculates the gradient and intercerpt of a lineal regression.
        @param ref: A 2-D Matrix with the refference data points.
        @type ref: 2-D Array list.
        @param iput: A 2-D Matrix with the input data points.
        @type iput: 2-D Array list.
        @return: x's axis gradient, x's axis intercept, y's axis gradient, y's axis intercept.
        @rtype: Float,Float,Float,Float.
        """

        gradientx, interceptx, r_value, p_value, std_err = stats.linregress(ref[:,0],iput[:,0])
        gradienty, intercepty, r_valuey, p_valuey, std_erry = stats.linregress(ref[:,1],iput[:,1])
        return gradientx,interceptx,gradienty,intercepty

    def getResiduals(self,xyfit):
        """
        Calculate the residuals for each point.
        @param xyfit: Fitted data values.
        @type xyfit: Float Points Array.
        @return: Array list with the residual values.
        @rtype: Array list. 
        """

        res=[]
        for i in xrange(self.n):
            res+=[[self.xyin[i,0]-xyfit[i,0],self.xyin[i,1]-xyfit[i,1]]]
        return np.array(res)

    def getRms(self,residuals):
        """
        Calculate the rms value for x y y axis.
        @param residuals: 2-D array with the residual values
        @type residuals: 2-D Array list.
        @return: x rms value, y rms value
        @rtype: Float,Float
        """

        xrms=0
        yrms=0
        for i in xrange(self.n):
            xrms+=residuals[i,0]**2
            yrms+=residuals[i,1]**2
        xrms=xrms/self.n
        yrms=yrms/self.n
        xrms=np.sqrt(xrms)
        yrms=np.sqrt(yrms)
        return xrms,yrms

    def RADTODEG(self,rad):
        """
        Transforms an angle from Radians to Degrees.
        @param rad: Angle measured in Radians.
        @type rad: Angle in Radians.
        @return: Same angle, but measured in Degrees.
        @rtype: Angle in Degrees.
        """

        return (rad * 180. / np.pi)

    def DEGTORAD(self,deg):
        """
        Transforms an angle from Degrees to Radians
        @param deg: Angle measured in Degrees
        @type deg: Angle in Degrees
        @return: Same angle but measured in Radians
        @rtype: Angle in Radians
        """

        return (deg * np.pi / 180.)

    def buildFit(self,P,Q):
        """
        Calculate the rotation angle.
        @param P: Array calculated as dot(NLA.inverse(M),U). It will be something like: P = np.array([-0.434589, -0.893084, 285.420816])
        @type P: Array
        @param Q: Array calculated as dot(NLA.inverse(M),V). It will be something like: Q = np.array([0.907435, -0.433864, 45.553862])
        @return: The rotation angle, and a factor.
        @rtype: Angle in radians, Angle in degrees, Factor
        """
        det = P[0]*Q[1] - P[1]*Q[0]
        if det > 0:
            p=1
        else:
            p=-1
        theta = np.arctan2(P[1] - p*Q[0], p*P[0] + Q[1])
        theta_deg = self.RADTODEG(theta) % 360.0
        return theta,theta_deg,p

    def accuMatrix(self):
        """
        Set up the products used for computing the fit
        @return: Matrix and two arrays  with the products.
        @rtype: Square Matrix, Array , Array
        """
        Sx = self.xyref[:,0].sum()
        Sy = self.xyref[:,1].sum()
        Su = self.xyin[:,0].sum()
        Sv = self.xyin[:,1].sum()
        Sux = dot(self.xyin[:,0],self.xyref[:,0])
        Svx = dot(self.xyin[:,1],self.xyref[:,0])
        Suy = dot(self.xyin[:,0],self.xyref[:,1])
        Svy = dot(self.xyin[:,1],self.xyref[:,1])
        Sxx = dot(self.xyref[:,0],self.xyref[:,0])
        Syy = dot(self.xyref[:,1],self.xyref[:,1])
        Sxy = dot(self.xyref[:,0],self.xyref[:,1])
        M = np.array([[Sx, Sy, self.n], [Sxx, Sxy, Sx], [Sxy, Syy, Sy]])
        U = np.array([Su,Sux,Suy])
        V = np.array([Sv,Svx,Svy])
        return M,U,V


    def shift(self):
        """
        Function that does a shift using Power Series polynomial.
        """
        (self.xmag,self.xshift,self.ymag,self.yshift)=self.getMag(self.xy,self.uv)
        diff_pts=[]
        for i in xrange(self.n):
            diff_pts += [[self.uv[i,0]-self.xy[i,0],self.uv[i,1]-self.xy[i,1]]]
        diff_pts= np.array(diff_pts)
        for i in xrange(self.n):
                self.xyfit+=[[self.xyref[i,0]+diff_pts[:,0].mean(),self.xyref[i,1]+diff_pts[:,1].mean()]]
        self.xshift,self.yshift=diff_pts[:,0].mean(),diff_pts[:,1].mean()
        self.xyfit=np.array(self.xyfit)
        self.residuals=self.getResiduals(self.xyfit)
        (self.xrms,self.yrms)=self.getRms(self.residuals)
        P = np.array([self.xmag,0,self.xshift])
        Q = np.array([0,self.ymag,self.yshift])
        theta,theta_deg,p=self.buildFit(P,Q)
        if self.xmag < 0:
            self.xmag=-self.xmag
        self.xrot=theta_deg
        self.yrot=theta_deg
        self.a=self.xshift
        self.d=self.yshift
        self.b=1
        self.e=0
        self.c=0
        self.f=1


    def xyscale(self):
        """
        Function that does a xyscale using Power Series polynomial.
        """
        (self.xmag,self.xshift,self.ymag,self.yshift)=self.getMag(self.xy,self.uv)
        P = np.array([self.xmag,0,self.xshift])
        Q = np.array([0,self.ymag,self.yshift])
        for i in xrange(self.n):
            self.xyfit+=[[self.xy[i,0]*self.xmag + self.xshift , self.xy[i,1]*self.ymag + self.yshift]]
        self.xyfit=np.array(self.xyfit)
        self.residuals=self.getResiduals(self.xyfit)
        (self.xrms,self.yrms)=self.getRms(self.residuals)
        theta,theta_deg,p=self.buildFit(P,Q)
        if self.xmag < 0:
            self.xmag=-self.xmag
            self.xrot=self.RADTODEG(np.arccos(-np.cos(theta))) % 360.0
        self.yrot=theta_deg
        self.e=0
        self.c=0
        self.a=self.xshift
        self.d=self.yshift
        self.b=self.xmag
        self.f=self.ymag


    def rotate(self):
        """
        Function that does a rotate using Power Series polynomial.
        """
        M,U,V=self.accuMatrix()
        P = dot(NLA.inv(M),U)
        Q = dot(NLA.inv(M),V)
        theta,theta_deg,p=self.buildFit(P,Q)
        if p < 0 :
            self.f=np.cos(theta)
            self.b=-self.f
            self.e=np.sin(theta)
            self.c=self.e
        else:
            self.f=np.cos(theta)
            self.b=self.f
            self.e=-np.sin(theta)
            self.c=-self.e
        diff_pts=[]
        for i in xrange(self.n):
            diff_pts+=[[self.uv[i,0]-self.b*self.xy[i,0]-self.c*self.xy[i,1],self.uv[i,1]-self.e*self.xy[i,0]-self.f*self.xy[i,1]]]
        diff_pts=np.array(diff_pts)
        self.a=diff_pts[:,0].mean()
        self.d=diff_pts[:,1].mean()
        for i in xrange(self.n):
            self.xyfit+=[[a+b*self.xy[i,0]+c*self.xy[i,1],d+e*self.xy[i,0]+f*self.xy[i,1]]]
        self.xyfit=np.array(self.xyfit)
        self.residuals=self.getResiduals(self.xyfit)
        (self.xrms,self.yrms)=self.getRms(self.residuals)
        self.yrot=theta_deg
        self.xrot=self.yrot
        if p < 0 :
            self.xrot=self.xrot+180
            if 360 < self.xrot:
                self.xrot=self.xrot-360
        self.xshift,self.yshift,self.xmag,self.ymag=self.a,self.d,1,1
    

    def rscale(self):
        """
        Function that does a rscale using Power Series polynomial.
        """
        M,U,V=self.accuMatrix()
        P = dot(NLA.inv(M),U)
        Q = dot(NLA.inv(M),V)
        det = P[0]*Q[1] - P[1]*Q[0]
        theta,theta_deg,p = self.buildFit(P,Q)
        det=p*det
        mag= np.sqrt(det)
        self.yrot=theta_deg
        self.xrot=self.yrot
        if p < 0 :
            self.xrot=self.xrot+180
            if 360 < self.xrot:
                self.xrot=self.xrot-360
        cthetax=np.cos(self.DEGTORAD(self.xrot))
        sthetax=np.sin(self.DEGTORAD(self.xrot))
        cthetay=np.cos(self.DEGTORAD(self.yrot))
        sthetay=np.sin(self.DEGTORAD(self.yrot))
        self.b=mag*cthetax
        self.c=mag*sthetay
        self.e=-mag*sthetax
        self.f=mag*cthetay
        diff_pts=[]
        for i in xrange(self.n):
            diff_pts+=[[self.uv[i,0]-self.b*self.xy[i,0]-self.c*self.xy[i,1],self.uv[i,1]-self.e*self.xy[i,0]-self.f*self.xy[i,1]]]
        diff_pts=np.array(diff_pts)
        self.a=diff_pts[:,0].mean()
        self.d=diff_pts[:,1].mean()
        for i in xrange(self.n):
            self.xyfit+=[[self.a+self.b*self.xy[i,0]+self.c*self.xy[i,1],self.d+self.e*self.xy[i,0]+self.f*self.xy[i,1]]]
        self.xyfit=np.array(self.xyfit)
        self.residuals=self.getResiduals(self.xyfit)
        (self.xrms,self.yrms)=self.getRms(self.residuals)
        self.xshift,self.yshift,self.xmag,self.ymag=self.a,self.d,mag,mag


    def rxyscale(self):
        """
        Function that does a rxyscale using Power Series polynomial.
        """
        M,U,V=self.accuMatrix()
        P = dot(NLA.inv(M),U)
        Q = dot(NLA.inv(M),V)
        theta,theta_deg,p=self.buildFit(P,Q)
        det = P[0]*Q[1] - P[1]*Q[0]
        det=p*det
        avg_scale = (((p*P[0]+Q[1])*np.cos(theta)) + ((P[1] - p*Q[0])*np.sin(theta)) )/2
        alpha = np.arcsin( (-p*P[0]*np.sin(theta)) - (p*Q[0]*np.cos(theta)))/(2*avg_scale)
        fact = ( ((p*P[0] - Q[1])*np.cos(theta)) - ((P[1]+p*Q[0])*np.sin(theta)))/(2*np.cos(alpha))
        scale_x = avg_scale + fact
        scale_y = avg_scale - fact
        self.xshift=P[2]
        self.yshift=Q[2]
        self.xmag=scale_x
        self.ymag=scale_y
        self.a=P[2]
        self.b=P[0]
        self.c=P[1]
        self.d=Q[2]
        self.e=Q[0]
        self.f=Q[1]
        self.yrot=theta_deg
        self.xrot=theta_deg
        if p < 0 :
            self.xrot=self.xrot+180
            if 360 < self.xrot:
                self.xrot=self.xrot-360
        for i in xrange(self.n):
            self.xyfit+=[[self.a+self.b*self.xy[i,0]+self.c*self.xy[i,1],self.d+self.e*self.xy[i,0]+self.f*self.xy[i,1]]]
        self.xyfit=np.array(self.xyfit)
        self.residuals=self.getResiduals(self.xyfit)
        (self.xrms,self.yrms)=self.getRms(self.residuals)


    def fxyr(self):
        """
        Function that does a fxyr using Power Series polynomial.
        """
        #this procedure is developed only for order= default and terms=none|half

        M,U,V=self.accuMatrix()
        P = dot(NLA.inv(M),U)
        Q = dot(NLA.inv(M),V)
        theta,theta_deg,p = self.buildFit(P,Q)
        avg_scale = (((p*P[0]+Q[1])*np.cos(theta)) + \
                    ((P[1] - p*Q[0])*np.sin(theta)) )/2
        alpha = np.arcsin( (-p*P[0]*np.sin(theta)) - \
                    (p*Q[0]*np.cos(theta)))/(2*avg_scale)
        fact = ( ((p*P[0] - Q[1])*np.cos(theta)) - \
                ((P[1]+p*Q[0])*np.sin(theta)))/(2*np.cos(alpha))

        scale_x = avg_scale + fact
        scale_y = avg_scale - fact
        self.xshift=P[2]
        self.yshift=Q[2]
        self.xmag=scale_x
        self.ymag=scale_y
        self.a=P[2]
        self.b=P[0]
        self.c=P[1]
        self.d=Q[2]
        self.e=Q[0]
        self.f=Q[1]
        #xmag => tg(phi)=sin(phi)/cos(phi)=-e/b => phi=360+ RADEG(atan(-e/b))
        #it has to be improved
        temp=-self.e/self.b
        temp=mt.atan(temp)
        temp=self.RADTODEG(temp)
        if 360 < temp :
            temp=temp-360
        elif temp < 0 and -1 < temp:
            temp = 360+temp
        elif temp <-1:
            temp = 180 + temp
        self.xrot=temp
        #ymag => phi=360+ RADEG(atan(c/f))
        temp=-self.c/self.f
        temp=mt.atan(temp)
        temp=self.RADTODEG(temp)
        if 360 < temp :
            temp=temp+360
        elif temp < 0:
            temp = 360-temp
        else:
            temp=360-temp
        self.yrot=temp

        for i in xrange(self.n):
            self.xyfit += [[self.a + self.b*self.xyref[i,0] + 
                          self.c*self.xyref[i,1],self.d + \
                          self.e*self.xyref[i,0] + self.f*self.xyref[i,1]]]

        self.xyfit = np.array(self.xyfit)
        self.residuals = self.getResiduals(self.xyfit)
        (self.xrms,self.yrms) = self.getRms(self.residuals)

    
    def getLambda(self) :
        """
        Procedure that creates a lambda function capable to transform the ref image into the in image. Works only for 1024 x 1024 images
        @return: Lambda function that get as input an array with the data from the ref image.
        @rtype: Function
        """
        #def mkfit(im):
        #    g2=np.zeros((1024,1024))
        #    for y in xrange(im.shape[0]):
        #        for x in xrange(im.shape[1]):
        #            xn = self.a + self.b*x + self.c*y
        #            yn = self.d + self.e*x + self.f*y
        #            if xn <0 or yn <0: continue
        #            if xn < 1024 and yn < 1024:
        #                # yn, xn are floating and they get truncated to
        #                # integer. It would be better to use interpolation.
        #                g2[yn,xn] = im[y,x]
        #    g2=np.array(g2)
        #    return g2
        def mkfit(im):
            sz = im.shape
            g2 = np.zeros(sz)
            print self.a , self.b , self.c , self.d , self.e , self.f
            for y in xrange(sz[0]):
                for x in xrange(sz[1]):
                    xn = self.a + self.b*x + self.c*y
                    yn = self.d + self.e*x + self.f*y
                    if xn <0 or yn <0: continue
                    if xn < 1024 and yn < 1024:
                        # yn, xn are floating and they get truncated to
                        # integer. It would be better to use interpolation.
                        g2[yn,xn] = im[y,x]
            g2=np.array(g2)
            return g2
        return lambda x: mkfit(x)


