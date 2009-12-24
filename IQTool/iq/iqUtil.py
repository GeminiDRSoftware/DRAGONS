import time
from utils import filesystem
import numpy as np
from pyraf import iraf
import detectSources
import pyfits as pf


"""This file contains the following utilities:

    convPars(fitPars, pixelscale,stampPars)
    fitfunction(stampArray, function, positionCoords, outFile, pixelscale, stampPars)
    sigmaClip(Pars, outFile, sigma=3, verbose=True, niters=2)
    removeNeighbors(xyArray, narcsec=5, pixelscale=1, crowded=True)
    printPars(clipobsPars, verbose)
    writePars(Pars, outFile, function)
    makeResiduals(scidata, fitPars, ReturnModel, stampPars)
    iqMark(frame, fitPars, color)

"""
#---------------------------------------------------------------------------
def pyDaoFind(filename, scidata, pixelscale,
              frame=1, debug=False, pymark=True,
              display=True, imageSigma='default',
	      saturation=65000, qa=False):
    
    xyArray = []

    if display:
        iraf.set(stdimage='imtgmos')
        iraf.display (filename, frame=frame)

    # mask out stars to get a good measure of background deviation for daofind
    if imageSigma == 'default': 
        maskeImage = starMask(scidata)
        imageSigma = maskedImage.std()

    # use default FWHM of 0.8 arcsec
    daoFWHM = 0.8/pixelscale
    
    maxObjects = 100
    # Epic kluge
    exten = int( filename.split( '[' )[1].split( ']' )[0] )
    #print "IU40:", imageSigma, daoFWHM, saturation
    st = time.time()
    if qa:
        # I will be the first to admit this is very unfriendly and kluged. Hopefully, I will have enough
        # time to improve it
        ratio = 5
        currentGrid = 0
        searchPattern = [12,13,11,7,17,6,8,16,18]
        gridSize = len(searchPattern)
        height, width = scidata.shape
        grid_width = width / ratio
        grid_height = height / ratio
        while len(xyArray) <= maxObjects and currentGrid < gridSize:
            grid = searchPattern[currentGrid]
            xoffset = (grid % ratio) * grid_width
            yoffset = (grid / ratio) * grid_height
            xyArray += detectSources.detSources( filename, sigma=imageSigma, threshold=3.0, fwhm=daoFWHM,
                                                 window=[(xoffset,yoffset,grid_width,grid_height)], exts=exten )
            currentGrid += 1
    
    else:
        xyArray = detectSources.detSources( filename, sigma=imageSigma, threshold=2.0, fwhm=daoFWHM, exts=exten )
    et = time.time()
    print 'IU44 - DAOFIND TIME:', (et-st)

    if pymark:
        filesystem.deleteFile('newcoofile2.temp')
        newcoofile2 = file('newcoofile2.temp', 'w')
    
        # mark objects found with daofind with blue 'X'
        for xy in xyArray:
            tmpcoo2 = '%5d%6d\n'%(xy[0]+1,xy[1]+1)
            newcoofile2.write(tmpcoo2)
    
        newcoofile2.close()
        iraf.tvmark(frame=frame,coords='newcoofile2.temp',
            mark='cross', color=206, pointsize=6)
        filesystem.deleteFile('newcoofile2.temp')

    if debug: 
        #print 'PYEXAM - DAOFIND outputs: '+str(xyArray)
        print "No. of Objects:", len(xyArray)
    return xyArray
#---------------------------------------------------------------------------

def edgeCheck(scidata, xyArray, npixaway=150, debug=False, xmin=None,
xmax=None):
    """mask out all pixels greater than 2.5 sigma

    @param scidata: science data array, containing only the object to be fit
    @type scidata: numpy array

    @param xyArray: list of (x,y) positions of objects
    @type xyArray: array

    @param npixaway: number of pix from edges to be removed from list
    @type npixaway: int or float

    """
    ymax, xmaxdum = scidata.shape
    ymin, xmindum = 0,0

    if xmin == None: xmin = xmindum
    if xmax == None: xmax = xmaxdum

    # first, make sure object in data
    i = 0
    while i <= (len(xyArray)-1):
        if xyArray[i][0] < xmin or xyArray[i][0] > xmax:
            if debug: print '# EDGECHECK - Removed ',xyArray[i],'\
outside x-range'
            xyArray.pop(i)
        elif xyArray[i][1] < ymin or xyArray[i][1] > ymax:
            if debug: print '# EDGECHECK - Removed ',xyArray[i],'\
outside y-range'
            xyArray.pop(i)
        elif abs(xmax - xyArray[i][0]) < npixaway:
            if debug: print '# EDGECHECK - Removed ',xyArray[i],' too\
close to upper x-edge'
            xyArray.pop(i)
        elif abs(ymax - xyArray[i][1]) < npixaway:
            if debug: print '# EDGECHECK - Removed ',xyArray[i],' too\
close to upper y-edge'
            xyArray.pop(i)
        elif abs(xyArray[i][0] - xmin) < npixaway:
            if debug: print '# EDGECHECK - Removed ',xyArray[i],' too\
close to lower x-edge'
            xyArray.pop(i)
        elif abs(xyArray[i][1] - ymin) < npixaway:
            if debug: print '# EDGECHECK - Removed ',xyArray[i],' too\
close to lower y-edge'
            xyArray.pop(i)
        i+=1

    return xyArray
#---------------------------------------------------------------------------
def starMask(scidata):
    """mask out all pixels greater than 2.5 sigma
    
    @param scidata: science data array, containing only the object to be fit
    @type scidata: numpy array
    """
    fim = []
    stars = []
    
    fim = scidata * 1.
    stars = np.where(fim > (1*scidata.std() + scidata.mean()))
    fim[stars] = scidata.mean()
    #nd.display(fim, frame=3)
    
    outside = np.where(fim < (1*scidata.std() - scidata.mean()))
    fim[outside] = scidata.mean()

    #nd.display(scidata, frame=2)
    #nd.display(fim, frame=4)
    return fim
#---------------------------------------------------------------------------

def iqmark(frame, fitPars, color):
    """tvmark iraf.display images with x,y and fwhm as radii

    @param frame: frame number for marking
    @type frame: int

    @param fitPars: least squares fit parameters
    @type fitPars: dict

    @param color: color code for mark
    @type color: int
    """
    filesystem.deleteFile('newcoofile.temp')
  
    newcoofile = file('newcoofile.temp','w')
    tmpcoo = '%5d%6d\n'%(fitPars['CooX'], fitPars['CooY'])
    newcoofile.write(tmpcoo)
    newcoofile.close()
    iraf.tvmark(frame=frame,coords='newcoofile.temp',
       mark='circle', radii=fitPars['FWHMpix'], color=color)    
    filesystem.deleteFile('newcoofile.temp')
 
#---------------------------------------------------------------------------
def makeResiduals(scidata, fitPars, returnModel, stampPars):
    """Create stamp array using fit parameters and their functions then subtract from original image

    @param scidata: science data array, containing only the object to be fit
    @type scidata: numpy array

    @param fitPars: function fit parameters 
    @type fitPars: dict

    @param returnModel: function from gauss or moffat fit 
    @type returnModel: function

    @param stampArray: science data array, ideally containing only the object to be fit
    @type stampArray: numpy array

    @return: subtract is a science data array with the object fit residuals subtracted
    @rtype: array
    """          
    if fitPars['Beta'] == 1:
       fxn = returnModel((0, fitPars['Peak'], fitPars['Cy'], fitPars['Cx'], \
                   fitPars['Wy'], fitPars['Wx'], fitPars['Theta']))
       
    else:
       fxn = returnModel((0, fitPars['Peak'], fitPars['Cy'], fitPars['Cx'], \
                      fitPars['Wy'], fitPars['Wx'], fitPars['Theta'], \
                      fitPars['Beta']))      
   
    stampArray = fxn(*np.indices((stampPars[3]-stampPars[2], \
                         stampPars[1]-stampPars[0])))
   
    imageDim = list(scidata.shape) # Gives [y,x] dimensions
    residualArray = np.zeros(imageDim)
    residualArray[stampPars[2]:stampPars[3],stampPars[0]:stampPars[1]] = stampArray
 
    subtract = scidata - residualArray

    return subtract



#---------------------------------------------------------------------------
def writePars(Pars, outFile, function):
    """writes lovely formatted string of parameters to outFile object

    @param Pars: parameters to be printed, this will be the full function fits and obsPars
    @type Pars: dict

    @param outFile: opened file object
    @type outFile: string

    @param function: currently printed as first column of each row of Pars
    @type function: string
    
    """
    #write pars to file
    outstring = '%10s%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%5d\n'%(function,
             Pars['Cx'], Pars['Cy'], Pars['Bg'], Pars['Peak'], Pars['Wx'], Pars['Wy'],
             Pars['CooX'], Pars['CooY'], Pars['Ellip'], Pars['FWHMpix'],
             Pars['FWHMarcsec'], Pars['Theta'], Pars['PAdeg'], Pars['FWHMx'], Pars['FWHMy'], Pars['Beta'], Pars['Frame'])
    outFile.write(outstring)
#---------------------------------------------------------------------------

def printPars(clipobsPars, verbose):
    """prints lovely formatted string of parameters to screen

    @param clipobsPars: parameters to be printed
    @type clipobsPars: list

    @param verbose: actually print the parameters?
    @type verbose: Boolean

    """
    
    if verbose == True:
       print '%10s%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f'% (clipobsPars) 

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
    else: #FWHMx == FWHMy
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

#---------------------------------------------------------------------------

def sigmaClip(Pars, outFile, sigma=2.3, verbose=True, nIters=4, garbageStat=False):
    """Median clip list of pars

    @param Pars: ObsPars from fit
    @type Pars: dict

    @param outFile: opened file object
    @type outFile: string    

    @param sigma: number of standard deviations away from the median
    @type sigma: int

    @param verbose: warnings will be printed
    @type verbose: Boolean

    @param nIters: number of times sigma clipping is iterated through parameter list
    @type nIters: int

    @return: Pars, EllMean, EllSigma, FWHMMean, FWHMSigma -- new clipped parlists and mean
                and std for each clippedlist
    @rtype: list
    """
    if len(Pars) <= 0:
        print '\n# PYEXAM - Cannot provide statistics with only no objects'
        return Pars, None, None, None, None

    if garbageStat:
        if len(Pars) < 3:
           print '\n# PYEXAM - WARNING, only one or two objects detected.'
           return Pars, None, None, None, None
    else:
        if len(Pars) < 2:
           print '\n# PYEXAM - Cannot provide statistics with only one object'
           return Pars, None, None, None, None
    
        if len(Pars) < 3:
           print '\n# PYEXAM - Cannot perform reasonable sigma clipping with less than three objects'
           return Pars, None, None, None, None
         
    ell = []   
    fwhm = []

    j = 1
    k = 1
    nparams = 2
    while j <= nIters:
        while k <= nparams:
            ell = []
            fwhm = []
            for star in Pars:
                fwhm.append(star.get('FWHMarcsec'))          
                ell.append(star.get('Ellip'))
            
            allPars = [fwhm, ell]
            key = ''
            
            i = 0
            
            copyallPars = allPars[:]
            
            for parlist in copyallPars:
                parlist.sort()
                tempMedian = parlist[len(parlist)/2] 
                tempSigma = np.array(parlist).std()
                lower = tempMedian - sigma*tempSigma
                upper = tempMedian + sigma*tempSigma
                
                cparlist = parlist[:]
                
                for par in cparlist:
                    if par < lower or par > upper:
                        if i == 0 : key='FWHMarcsec'
                        elif i == 1: key='Ellip'
                        for star in Pars:
                            if star[key]==par:
                                Pars.remove(star)
                                parlist.remove(par)
                                #replace old parlist in allPars with this one
                                allPars[i] = parlist
                                if verbose:
                                    pstr = '# SIGMACLIP - removed object with '+str(key)+" = "+str(par)
                                    print pstr
                i += 1
            k += 1
        j += 1

    FWHMMean = np.array(allPars[0]).mean()
    FWHMSigma = np.array(allPars[0]).std()
    EllMean = np.array(allPars[1]).mean()
    EllSigma = np.array(allPars[1]).std()
 
    return Pars, EllMean, EllSigma, FWHMMean, FWHMSigma

#---------------------------------------------------------------------------

def removeNeighbors(xyArray, npixapart=5, crowded=True, debug=False):
    """
    Remove neighbors from a list of x,y positions within narcsec of each other.

    @param xyArray: list of (x,y) positions of objects
    @type xyArray: array

    @param npixapart: number of pix apart for neighbors to be removed from list.
    @type npixapart: int or float

    @param pixelscale: instrument pixelscale in arcsec/pix
    @type pixelscale: float or int

    @param crowded: (OBSOLETE) if crowded=False then check remove up to triplets,
    if crowded.
    @type crowded: Boolean

    @return: xyArray list of neighbor removed list of (x,y) positions of
    objects.
    @rtype: array
    """

    xyArray.sort()
    xyArrayForRemoval = []
    xyRemoveFlag = False
    j = 0
    while j < (len(xyArray)-1):
        i = j + 1
        while i < len(xyArray):
            diffx = xyArray[j][0] - xyArray[i][0]
            if abs(diffx) < npixapart:
                diffy = xyArray[j][1] - xyArray[i][1]
                if abs(diffy) < npixapart:
                    if debug: print '\n# REMOVENEIGHBORS - Removed neighbors!', xyArray[j], xyArray[i]
                    if not xyRemoveFlag:
                        xyRemoveFlag = True
                        xyArrayForRemoval.append(j)
                    xyArrayForRemoval.append(i)
            else:
                break

            i += 1

        if xyRemoveFlag:
            # The reverse() is to make sure the later index
            # neighbours are removed first as to not effect the
            # len and other earlier indexes.
            xyArrayForRemoval.reverse()
            for xyRemovePointIndex in xyArrayForRemoval:
                xyArray.pop(xyRemovePointIndex)
            xyArrayForRemoval = []
            xyRemoveFlag = False
            j = j - 1
      
        j = j + 1

    return xyArray

