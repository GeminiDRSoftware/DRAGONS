
import iqUtil
import numpy as np
from scipy.optimize import minpack


"""This file contains the following utilities:
    fitprofile(stampArray, function, positionCoords, outFile, pixelscale,
        stampPars)
    gauss2d (stampArray, pixelscale, intitialParams, stampPars, outFile)
    moffat2d (stampArray, pixelscale, intitialParams, stampPars, outFile)
    getApSize (pixelscale, boxSize)
    moments (scidata)
    makeStamp(imgArray, positionCoords, apertureSize, outFile)
    convPars(fitPars, pixelscale,stampPars)    
    """
#---------------------------------------------------------------------------

def fitprofile(stampArray, function, positionCoords, outFile, pixelscale, stampPars, debug, frame=1):
    """A wrapper for gauss2d and moffat2d fitting routines

    @param stampArray: science data array, ideally containing only the object to be fit
    @type stampArray: numpy array
    

    @param function: currently supported are 'gauss', 'moffat', or 'both'
    @type function: string

    @param postionCoords: (x,y) of object in image
    @type: list

    @param outFile: opened file object
    @type outFile: string    

    @param pixelscale: instrument pixelscale in arcsec/pix
    @type pixelscale: float or int

    @param stampPars: stampsize to convert stamp x,y to image x,y in convPars format: [xlow,xhigh,ylow,yhigh]
    @type stampPars: list

    @return: gPars, gReturnModel, mPars, mReturnModel, if moffat/gauss not fitting or didn't converge,
           returns None
    @rtype: dict, function, dict, function
    """
  
    gPars = None
    mPars = None
    gReturnModel = None
    mReturnModel = None
    if stampArray == None: return None, None, None, None

    initialParams = inpars(stampArray, pixelscale)    

    if function == 'gauss' or function == 'both':
       gPars = {}
       gReturnModel = {}
       gPars, gReturnModel = gauss2d(stampArray, pixelscale, initialParams, stampPars, outFile, debug, frame=frame) 
    
    if function == 'moffat' or function == 'both':
       mPars = {}
       mReturnModel = {}
       mPars, mReturnModel = moffat2d(stampArray, pixelscale, initialParams, stampPars, outFile, debug, frame=frame)
                 
    elif function != 'gauss':
      print "# PYEXAM - Function "+function+" not recognized, please put either gauss/moffat/both"
 
    return gPars, gReturnModel, mPars, mReturnModel


#---------------------------------------------------------------------------
def gauss2d(stampArray,pixelscale,initialParams,stampPars, outFile, debug, frame):
    """Least squares fit a Gaussian function to a science array.

    @param stampArray: science data array, ideally containing only the object to be fit
    @type stampArray: numpy array

    @param pixelscale: instrument pixelscale in arcsec/pix
    @type pixelscale: float or int

    @param intialParams: initial parameters for least squares fitting routine
    @type initialParams: list

    @param stampPars: stampsize to convert stamp x,y to image x,y in convPars format: [xlow,xhigh,ylow,yhigh]
    @type stampPars: list

    @param outFile: opened file object
    @type outFile: string

    @return: GaussFitPars, returnModel
    @rtype: dict, function
    """          

    (bg,peak,cx,cy,wx,wy,theta) = initialParams
           
    def gaussian2dModel(bg, peak, cx, cy, wx, wy, theta):
        wx = float(wx)
        wy = float(wy)
        return lambda x,y: bg+peak*np.exp(-((((x-cx)*np.cos(theta)+(y-cy)*np.sin(theta))/wx)**2+(((x-cx)*np.sin(theta)-(y-cy)*np.cos(theta))/wy)**2)/2)
    # The next three lines make the least squares fit
    returnModel = lambda params: gaussian2dModel(*params)
    errorfunction = lambda inputParams:np.ravel(gaussian2dModel(*inputParams)(*np.indices(stampArray.shape)) - stampArray)
    (Bg, Peak, Cx, Cy, Wx, Wy, Theta),success = minpack.leastsq(errorfunction, initialParams, maxfev=100, warning=False)

    # Create fit parameters dictionary
    if success < 4:
        #stamp array has (y,x) instead of (x,y), need to switch everything
        Wx, Wy, Cx, Cy = Wy, Wx, Cy, Cx
        Theta = Theta-180
                
        GaussFitPars = {'Bg':Bg,'Peak':Peak,'Cx':Cx,'Cy':Cy,
                 'Wx':Wx,'Wy':Wy,'Theta':Theta, 'Beta':1}

        # Convert the fit parameters to observables like location, fwhm, ellipticity and PA
        fitPars=(Bg, Peak, Cx, Cy, Wx, Wy, Theta)        
        CooX,CooY,FWHMx,FWHMy,FWHMpix,FWHMarcsec,PAdeg,ellipticity = iqUtil.convPars(fitPars, pixelscale,stampPars)

        # Add observables to dictionary and specify that each variable was from Gauss fit
        GaussObsPars = {'CooX':CooX, 'CooY':CooY, 'FWHMx':FWHMx, 'FWHMy':FWHMy,
                    'FWHMpix':FWHMpix, 'FWHMarcsec':FWHMarcsec, 'PAdeg':PAdeg,
                    'Ellip':ellipticity, 'Frame':frame}

        GaussFitPars.update(GaussObsPars)
    else:
        if debug: print ("# PYEXAM - gaussian fit did not converge at x="+str(Cx+stampPars[1]+1)+" y="+str(Cy+stampPars[3]+1))
        GaussFitPars = None

    return GaussFitPars, returnModel

#---------------------------------------------------------------------------
def moffat2d(stampArray,pixelscale,initialParams,stampPars,outFile, debug, frame):
    """Least square fit a Moffat function to a science array.

    @param stampArray: science data array, ideally containing only the object to be fit
    @type stampArray: numpy array

    @param pixelscale: instrument pixelscale in arcsec/pix
    @type pixelscale: float or int

    @param intialParams: initial parameters for least squares fitting routine
    @type initialParams: list

    @param stampPars: stampsize to convert stamp x,y to image x,y in convPars format: [xlow,xhigh,ylow,yhigh]
    @type stampPars: list

    @param outFile: opened file object
    @type outFile: string

    @return: MoffatFitPars, returnModel 
    @rtype: dict, function
    """          

    (bg,peak,cx,cy,wx,wy,theta) = initialParams
    beta = 2.5

    initialParams = (bg, peak, cx, cy, wx, wy, theta, beta)

    def moffat2dModel(bg, peak, cx, cy, wx, wy, theta, beta):
        wx = float(wx)
        wy = float(wy)
        return lambda x,y: bg + peak*(1+(((x-cx)*np.cos(theta)+(y-cy)*np.sin(theta))/wx)**2 + (((x-cx)*np.sin(theta)-(y-cy)*np.cos(theta))/wy)**2)**(-beta)

    # The next three lines make the least squares fit
    returnModel = lambda params: moffat2dModel(*params)
    errorfunction = lambda inputParams:np.ravel(moffat2dModel(*inputParams)(*np.indices(stampArray.shape)) - stampArray)
    (Bg, Peak, Cx, Cy, Wx, Wy, Theta, Beta),success = minpack.leastsq(errorfunction,initialParams,warning=False,
                                                                      maxfev=100, col_deriv=1)

    # Create fit parameters dictionary
    if success < 4:
        #stamp array has (y,x) instead of (x,y), need to switch everything
        Wx, Wy, Cx, Cy = Wy, Wx, Cy, Cx
        Theta = Theta-180
        
        MoffatFitPars = {'Bg':Bg,'Peak':Peak,'Cx':Cx,'Cy':Cy, 'Wx':Wx,'Wy':Wy,'Theta':Theta,'Beta':Beta}

        # convert Moffat Wx to Gaussian Wx (sigma)
        Wx = Wx*np.sqrt(((2**(1/Beta)-1)/(2*np.log(2))))
        Wy = Wy*np.sqrt(((2**(1/Beta)-1)/(2*np.log(2))))

        # Convert the fit parameters to observables like location, fwhm, ellipticity and PA
        fitPars = (Bg, Peak, Cx, Cy, Wx, Wy, Theta)
    
        CooX, CooY, FWHMx, FWHMy, FWHMpix, FWHMarcsec, PAdeg, ellipticity = iqUtil.convPars(fitPars, pixelscale, stampPars)

        # Add observables to dictionary and specify that each variable was from Moffat fit
        MoffatObsPars = {'CooX':CooX, 'CooY':CooY, 'FWHMx':FWHMx, \
                 'FWHMy':FWHMy, 'FWHMpix':FWHMpix, 'FWHMarcsec':FWHMarcsec, \
                 'PAdeg':PAdeg, 'Ellip':ellipticity, 'Frame':frame}

        MoffatFitPars.update(MoffatObsPars)
    else:
        if debug: print ("# MOFFAT2D - moffat fit did not converge at x="+str(cx+stampPars[0]+1)+" y="+str(cy+stampPars[2]+1))
        MoffatFitPars = None
    
    return MoffatFitPars, returnModel

#---------------------------------------------------------------------------
def inpars(scidata, pixelscale):
    """Computes intial conditions for postage stamp fitting

    @param scidata: science data array, usually created with pyfits
    @type scidata: numpy array

    @param pixelscale: instrument pixelscale in arcsec/pix
    @type pixelscale: float or int

    @return: (bg, peak, x, y, wxx, wyy, theta) = background, peak, xcenter, ycenter, fwhmx, \
             fwhmy, theta
    @rtype: list
    """        

    peak = scidata.max() ## assume the peak pixel in the subarray corresponds with
                   ## the peak of the object in the subarray
    bg = scidata.mean()
    wxx = 0.7/pixelscale # assume 0.7 arcsec seeing as initial condition
    wyy = 0.7/pixelscale
    theta =0.
    x, y = scidata.shape
    x = x/2.
    y = y/2.

    ## Moffat fit Only ## how to estimate Beta of object?
    return bg, peak, x, y, wxx, wyy, theta

#---------------------------------------------------------------------------

def getApSize(pixelscale, boxSize):
    """Returns converted box size in pixels.

    @param pixelscale: instrument pixelscale in arcsec/pix
    @type pixelscale: float or int

    @param boxsize: size of stamp for fitting routines
    @type boxsize: float or int

    @return: apertureSize in pixels
    @rtype: float
    """    
    apertureSize=float(boxSize)/float(pixelscale)
    return apertureSize

#---------------------------------------------------------------------------

def makeStamp(scidata, positionCoords, apertureSize, outFile, debug):
    """create a postage stamp around positionCoords with box size apertureSize from imgArray

    @param scidata: science data array, containing only the object to be fit
    @type scidata: numpy array

    @param postionCoords: (x,y) of object in image
    @type: list

    @param apertureSize: stampsize in Pixels, stamp is a box of side 2xapertureSize
    @rtype: float

    @param outFile: opened file object
    @type outFile: string    

    @return: stampArray, stampPars
    @rtype: numpy array, list
    
    """       #____________this should be setStamp funtion
    objX,objY = positionCoords    
    imgYsize, imgXsize = scidata.shape ##shape returns (y-size, x-size)    
    ##high and low x-boundries of the subarray
    
    stampXlow, stampXhigh = int(round(objX-apertureSize)), int(round(objX+apertureSize)) 
    stampYlow, stampYhigh = int(round(objY-apertureSize)), int(round(objY+apertureSize))
    ##high and low y-boundries of the subarray
    if stampXlow > 0 and stampXhigh < imgXsize and stampYlow > 0 and stampYhigh < imgYsize:
       stampArray = scidata[stampYlow:stampYhigh,stampXlow:stampXhigh] #numpy indices y,x       
       stampPars=(stampXlow,stampXhigh,stampYlow, stampYhigh)

    else:
        if debug: print ("\n# MAKESTAMP - object at X="+str(objX)+" Y="+str(objY)+" makes a stamp outside of image \n")
        return None, None

    return stampArray, stampPars
    
    #-----------------------------------------------------------
    # Will be nice to implement circular box eventually
    #y, x = np.indices(f.shape, dtype=float32)
    #x = x - objX
    #y = y - objY
    #radius = np.sqrt(x**2 + y**2)
    #mask = radius < apertureSize
    #stampArray = mask * f
    #display (stampArray)
    #-----------------------------------------------------------
#---------------------------------------------------------------------------



def convPars(fitPars, pixelscale,stampPars):
    """Takes fit parameters output from Gaussian/Moffat fitting routines and converts them to \
    ellipticity, fwhm and pa

    @param fitPars: function fit parameters: Bg, Peak, Cx, Cy, Wx, Wy, Theta 
    @type fitPars: list

    @param pixelscale: instrument pixelscale in arcsec/pix
    @type pixelscale: float or int

    @param stampPars: stampsize to convert stamp x,y to image x,y --format: [xlow,xhigh,ylow,yhigh]
    @type stampPars: list

    @return: obsPars
    @rtype: list
    """          
       
    Bg, Peak, Cx, Cy, Wx, Wy, Theta = fitPars

    FWHMx = abs(2*np.sqrt(2*np.log(2))*Wx)
    FWHMy = abs(2*np.sqrt(2*np.log(2))*Wy)
    PAdeg = (Theta*(180/np.pi))
    PAdeg = PAdeg%360
                
    if FWHMy < FWHMx:
       ellip = 1 - FWHMy/FWHMx
       PAdeg = PAdeg
       FWHM = FWHMx
    elif FWHMx < FWHMy:
       ellip = 1 - FWHMx/FWHMy                    
       PAdeg = PAdeg-90 
       FWHM = FWHMy
    elif FWHMx == FWHMy:
       ellip = 0
       FWHM = FWHMx

    if PAdeg > 180:
       PAdeg=PAdeg-180

    if PAdeg < 0:
       PAdeg=PAdeg+180

    lowX,highX,lowY,highY = stampPars
    CooX, CooY= lowX+Cx+1,lowY+Cy+1
    #CooX, CooY= lowX+Cx,lowY+Cy
    FWHMarcsec = FWHM*pixelscale
    obsPars = (CooX, CooY, FWHMx, FWHMy, FWHM, FWHMarcsec, PAdeg,ellip)
    return obsPars
