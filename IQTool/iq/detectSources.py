from convolve import convolve2d
import math
from numpy import *
import os
import pyfits as pf
import time
#------------------------------------------------------------------------------ 
from utils import paramutil
#------------------------------------------------------------------------------ 
def detSources( image, outfile="", verbose=False, sigma=0.0, threshold=2.5, fwhm=5.5, 
                sharplim=[0.2,1.0], roundlim=[-1.0,1.0], window=None, exts=None, 
                timing=False, grid=False, rejection=None, ratio=None, drawWindows=False,
                dispFrame=1 ):
    """
    Performs similar to the source detecting algorithm 
    'http://idlastro.gsfc.nasa.gov/ftp/pro/idlphot/find.pro'.
    
    This code is heavily influenced by 'http://idlastro.gsfc.nasa.gov/ftp/pro/idlphot/find.pro'.
    'find.pro' was written by W. Landsman, STX  February, 1987.
    
    This code was converted to Python with areas re-written for optimization by:
    River Allen, Gemini Observatory, December 2009. riverallen@gmail.com
    
    
    Sources:
    [1] - W. Landsman. http://idlastro.gsfc.nasa.gov/ftp/pro/idlphot/find.pro
    
    @param image: The filename of the fits file. It must be in the format N2.fits[1] for the specific 
    extension. (i.e.) If you want to find objects only in the image extension [1], than you would pass N2.fits[1].
    @type filename: String
    
    @param outfile: The name of the file where the output will be written. By default output will not be written (ie if outfile
    is left as "", no output file is written).
    @type outfile: String
    
    @param verbose: Print out non-critical and debug information.
    @type verbose: Boolean
    
    @param sigma: The mean of the background value. If nothing is passed, detSources will run 
    background() to determine it.
    @type sigma: Number
    
    @param threshold: "Threshold intensity for a point source - should generally be 3 or 4 sigma 
    above background RMS"[1]. It was found that 2.5 works best for IQ source detection.
    @type threshold: Number
    
    @param fwhm: "FWHM to be used in the convolve filter"[1]. This ends up playing a factor in 
    determining the size of the kernel put through the gaussian convolve.
    @type fwhm: Number
    
    @param sharplim: "2 element vector giving low and high cutoff for the sharpness statistic (Default: [0.2,1.0] ).
    Change this default only if the stars have significantly larger or smaller concentration than a Gaussian"[1]
    @type sharplim: 2-Element List of Numbers
    
    @param roundlim: "2 element vector giving low and high cutoff for the roundness statistic (Default: [-1.0,1.0] ).
    Change this default only if the stars are significantly elongated."[1]
    @type roundlim: 2-Element List of Numbers
    
    @param window: Rectangle regions of the data to process. detSources will only look at the data within
    windows passed, if a window is passed. If no window is set, detSources will look at the entire image.
    Beware: small objects on the edges of the windows may not be detected.
    
    <pre>
    General Coordinate Form:
    ( x_offset, y_offset, width, height )
    
                     (x_offset + width, y_offset + height)
         __________ /
        |  Window  |
        |__________|
       /
    (x_offset, y_offset)
    
    Example:
    Window=[(0,0,200,200)] ~~ Looks at a window of size 200, 200 in bottom left corner
    Window=[(0,0,halfWidth,Height),(halfWidth,0,halfWidth,Height)] ~~ Splits the image in 2, divided vertically
        down the middle.
    </pre>
    @type window: List of 4 dimensional tuples or None  

    @param timing: If timing is set to true, the return type for detSources will be a tuple. The tuple 
    is of the form (xyArray, overalltime) where overalltime represents the time it took detSources to 
    run minus any displaying time. This feature is for engineering purposes.
    @type timing: Boolean
    
    @param grid: If no window is set, detSources will run the image in a grid. This is supposed to work in
    conjunction with rejection.
    @type grid: Boolean
    
    @param rejection: Rejection functions to be run on each grid point. See baseHeuristic() for an example.
    @type rejection: A list of rejection functions or None
    
    @param ratio: What the ratio or grid size should be. Ratio of 5 means the image will be split up into a 
    5x5 grid. Should be modified to take fixe grid size (50,50), for example.
    @type ratio: int
    
    @param drawWindows: If this is set to True, will attempt to draw the windows using iraf.tvmark().
    Beware: a ds9 must be running.
    @type drawWindows: Boolean
    
    @param dispFrame: This works in conjunction with drawWindows.
                debug=False, grid=False, rejection=None, ratio=None, drawWindows=False,
                dispFrame=1
    
    
    
    @return: A List of centroids. For example:
    
    [[ 626.66661222,  178.89720247],
     [  718.1319315 ,  2265.69332291],
     [ 783.03009601,   13.21621043],
     [ 1161.89652591,  2149.35972066],
     [ 1228.65067586,  1873.15018455],
     [ 1339.96915669,   725.79570466],
     [ 1477.96348539,  1107.85307289],
     [ 1485.17058871,  2059.1712877 ],
     [ 1501.959992  ,   227.32708114],
     [ 2003.10937888,   572.89806682],
     [ 2217.95000197,   763.01713875],
     [ 2407.5780915 ,  2018.30400873]]
     
    @rtype: 2-D List.
    """   
    
    #===========================================================================
    # Parameter Checking
    #===========================================================================
#    image = paramutil.checkParam( image, str, "" )
    if image == "":
        raise "daoFind requires an image file."
    
    imageName, ext = paramutil.checkFileFitExtension( image )
    if verbose:
        print "Opening and Loading:", imageName
    
    hdu = pf.open( imageName )
    
    if window is not None:
        if type(window) == tuple:
            window = [window]
        elif type(window) == list:
            pass
        else:
            raise "'window' must be a tuple of length 4, or a list of tuples length 4."
            
        for wind in window:
            if type(wind) == tuple:
                if len(wind) == 4:
                    continue
                else:
                    raise 'A window tuple has incorrect information, %s, require x,y,width,height' %(str(wind))
            else:
                raise 'The window list contains a non-tuple. %s' %(str(wind))
        
    if type( exts ) != int and exts is not None:
        raise 'exts must be int or None.' 
    
    
    
        
#    outfile = paramutil.checkParam( outfile, str, "" )
    
    writeOutFlag = False
    if outfile != "":
        writeOutFlag = True
    
#    fwhm = paramutil.checkParam( fwhm, type(0.0), 5.5, 0.0 )
#    verbose = paramutil.checkParam( verbose, bool, False )    
        
    if len(sharplim) < 2:
        raise "Sharplim parameter requires 2 num elements. (i.e. [0.2,1.0])"
    if len(roundlim) < 2:
        raise "Roundlim parameter requires 2 num elements. (i.e. [-1.0,1.0])"
    
    if verbose:
        print "Opened and loaded."
    #------------------------------------------------------------------------------ 
    #===========================================================================
    # Setup
    #===========================================================================
    ost = time.time()
    maxConvSize = 13     #Maximum size of convolution box in pixels 
        
    radius = maximum(0.637 * fwhm, 2.001)             #Radius is 1.5 sigma
    radiusSQ = radius ** 2
    kernelHalfDimension = minimum(array(radius, copy=0).astype(int32), (maxConvSize - 1) / 2)
    kernelDimension = 2 * kernelHalfDimension + 1    # Dimension of the kernel or "convolution box"
    
    sigSQ = (fwhm / 2.35482) ** 2
    
    # Mask identifies valid pixels in convolution box 
    mask = zeros([kernelDimension, kernelDimension], int8)
    # g will contain Gaussian convolution kernel
    gauss = zeros([kernelDimension, kernelDimension], float32)
    
    row2 = (arange(kernelDimension) - kernelHalfDimension) ** 2
    
    for i in arange(0, (kernelHalfDimension)+(1)):
        temp = row2 + i ** 2
        gauss[kernelHalfDimension - i] = temp
        gauss[kernelHalfDimension + i] = temp
    
    
    mask = array(gauss <= radiusSQ, copy=0).astype(int32)     #MASK is complementary to SKIP in Stetson's Fortran
    good = where(ravel(mask))[0]  #Value of c are now equal to distance to center
    pixels = good.size
    
    # Compute quantities for centroid computations that can be used for all stars
    gauss = exp(-0.5 * gauss / sigSQ)
    
    """
     In fitting Gaussians to the marginal sums, pixels will arbitrarily be
     assigned weights ranging from unity at the corners of the box to
     kernelHalfDimension^2 at the center (e.g. if kernelDimension = 5 or 7, the weights will be
    
                                     1   2   3   4   3   2   1
          1   2   3   2   1          2   4   6   8   6   4   2
          2   4   6   4   2          3   6   9  12   9   6   3
          3   6   9   6   3          4   8  12  16  12   8   4
          2   4   6   4   2          3   6   9  12   9   6   3
          1   2   3   2   1          2   4   6   8   6   4   2
                                     1   2   3   4   3   2   1
    
     respectively).  This is done to desensitize the derived parameters to
     possible neighboring, brighter stars.[1]
    """
    
    xwt = zeros([kernelDimension, kernelDimension], float32)
    wt = kernelHalfDimension - abs(arange(kernelDimension).astype(float32) - kernelHalfDimension) + 1
    for i in arange(0, kernelDimension):
        xwt[i] = wt
    
    ywt = transpose(xwt)
    sgx = sum(gauss * xwt, 1)
    sumOfWt = sum(wt)
    
    sgy = sum(gauss * ywt, 0)
    sumgx = sum(wt * sgy)
    sumgy = sum(wt * sgx)
    sumgsqy = sum(wt * sgy * sgy)
    sumgsqx = sum(wt * sgx * sgx)
    vec = kernelHalfDimension - arange(kernelDimension).astype(float32)
    
    dgdx = sgy * vec
    dgdy = sgx * vec
    sdgdxs = sum(wt * dgdx ** 2)
    sdgdx = sum(wt * dgdx)
    sdgdys = sum(wt * dgdy ** 2)
    sdgdy = sum(wt * dgdy)
    sgdgdx = sum(wt * sgy * dgdx)
    sgdgdy = sum(wt * sgx * dgdy)
    
    kernel = gauss * mask          #Convolution kernel now in c      
    sumc = sum(kernel)
    sumcsq = sum(kernel ** 2) - (sumc ** 2 / pixels)
    sumc = sumc / pixels
    
    # The reason for the flatten is because IDL and numpy treat statements like arr[index], where index 
    # is an array, differently. For example, arr.shape = (100,100), in IDL index=[400], arr[index]
    # would work. In numpy you need to flatten in order to get the arr[4][0] you want.
    kshape = kernel.shape
    kernel = kernel.flatten()
    kernel[good] = (kernel[good] - sumc) / sumcsq
    kernel.shape = kshape
    
    # Using row2 here is pretty confusing (From IDL code)
    # row2 will be something like: [1   2   3   2   1]
    c1 = exp(-.5 * row2 / sigSQ)
    sumc1 = sum(c1) / kernelDimension
    sumc1sq = sum(c1 ** 2) - sumc1
    c1 = (c1 - sumc1) / sumc1sq
    
    mask[kernelHalfDimension,kernelHalfDimension] = 0    # From now on we exclude the central pixel
        
    pixels = pixels - 1      # so the number of valid pixels is reduced by 1
    # What this operation looks like:
    # ravel(mask) = [0 0 1 1 1 0 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 ...]
    # where(ravel(mask)) = (array([ 2,  3,  4,  8,  9, 10, 11, 12, 14, ...]),)
    good = where(ravel(mask))[0]      # "good" identifies position of valid pixels
    
    # x and y coordinate of valid pixels 
    xx = (good % kernelDimension) - kernelHalfDimension
    
    # relative to the center
    yy = array(good / kernelDimension, copy=0).astype(int32) - kernelHalfDimension
    
    
    #------------------------------------------------------------------------------ 
    #===========================================================================
    # Extension and Window / Grid
    #===========================================================================
    
    xyArray = []
    outputLines = []
    
    
    if exts is None:
        # May want to include astrodata here to deal with
        # all 'SCI' extensions, etc.
        exts = 1

    
    sciData = hdu[exts].data
    
    if sigma <= 0.0:
        sigma = background( sciData )
        if verbose:
            print 'Estimated Background:', sigma
    
    hmin = sigma * threshold
    
    if window is None:
        # Make the window the entire image
        window = [(0,0,sciData.shape[1],sciData.shape[0])]
    
    if grid:
        ySciDim, xSciDim = sciData.shape
        xgridsize = int(xSciDim / ratio) 
        ygridsize = int(ySciDim / ratio)
        window = []
        for ypos in range(ratio):
            for xpos in range(ratio):
                window.append( (xpos * xgridsize, ypos * ygridsize, xgridsize, ygridsize) )
    
    
    
    drawtime = 0
    if drawWindows:
        drawtime = draw_windows( window, dispFrame, label=True)
    
    if rejection is None:
        rejection = []
    elif rejection is 'default':
        rejection = [baseHeuristic]
        
    windName = 0
    for wind in window:
        windName += 1
        subXYArray = []
        
        ##@@TODO check for negative values, check that dimensions don't violate overall dimensions.
        yoffset, xoffset, yDimension, xDimension = wind
        
        if verbose:
            print 'x,y,w,h', xoffset, yoffset, xDimension, yDimension
            print '='*50
            print 'W' + str(windName)
            print '='*50
        
        sciSection = sciData[xoffset:xoffset+xDimension,yoffset:yoffset+yDimension]
        
        #=======================================================================
        # Quickly determine if a window is worth processing
        #=======================================================================
        rejFlag = False
        
        for rejFunc in rejection:
            if rejFunc(sciSection, sigma, threshold):
                rejFlag = True
                break
        
        if rejFlag:
            # Reject
            continue
        
        #------------------------------------------------------------------------------
        #===========================================================================
        # Convolve
        #===========================================================================
        if verbose:
            print "Beginning convolution of image"
        
        st = time.time()
        
        h = convolve2d( sciSection, kernel )    # Convolve image with kernel
        
        et = time.time()
        if verbose:
            print 'Convole Time:', ( et-st )
    
        if not grid:
            h[0:kernelHalfDimension,:] = 0
            h[xDimension - kernelHalfDimension:xDimension,:] = 0
            h[:,0:kernelHalfDimension] = 0
            h[:,yDimension - kernelHalfDimension:yDimension] = 0
        
        if verbose:
            print "Finished convolution of image"
        
        #------------------------------------------------------------------------------ 
        #===========================================================================
        # Filter
        #===========================================================================
        offset = yy * xDimension + xx
        
        index = where(ravel(h >= hmin))[0]  # Valid image pixels are greater than hmin
        nfound = index.size

        if nfound > 0:             # Any maxima found?      
            h = h.flatten()
            for i in arange(pixels):
                # Needs to be changed
                try:
                    stars = where(ravel(h[index] >= h[index+ offset[i]]))[0]
                except:
                    break
                nfound = stars.size
                if nfound == 0:     # Do valid local maxima exist?
                    if verbose:
                        print "No objects found."
                    break
                index = index[stars]
            h.shape = (xDimension, yDimension)
            
            ix = index % yDimension               # X index of local maxima
            iy = index / yDimension               # Y index of local maxima
            ngood = index.size
        else:
            if verbose:
                print "No objects above hmin (%s) were found." %(str(hmin))
            continue
                
        #  Loop over star positions; compute statistics
        
        st = time.time()
        for i in arange(ngood):
            temp = array(sciSection[iy[i] - kernelHalfDimension:(iy[i] + kernelHalfDimension)+1,
                                 ix[i] - kernelHalfDimension:(ix[i] + kernelHalfDimension)+1])
            
            pixIntensity = h[iy[i],ix[i]]   # pixel intensity        
            
            #  Compute Sharpness statistic
            #@@FIXME: This should do proper checking...the issue is an out of range index with kernelhalf and temp
            # IndexError: index (3) out of range (0<=index<=0) in dimension 0
            try:
                sharp1 = (temp[kernelHalfDimension,kernelHalfDimension] - (sum(mask * temp)) / pixels) / pixIntensity
            except:
                continue
            
            if (sharp1 < sharplim[0]) or (sharp1 > sharplim[1]):   
                # Reject
                # not sharp enough?
                continue
            
            dx = sum(sum(temp, 1) * c1)
            dy = sum(sum(temp, 0) * c1)
            
            if (dx <= 0) or (dy <= 0):   
                # Reject
                continue
            
            around = 2 * (dx - dy) / (dx + dy)    # Roundness statistic
            
            # Reject if not within specified roundness boundaries.
            
            if (around < roundlim[0]) or (around > roundlim[1]):   
                # Reject
                continue
            
            """
             Centroid computation:   The centroid computation was modified in Mar 2008 and
             now differs from DAOPHOT which multiplies the correction dx by 1/(1+abs(dx)).
             The DAOPHOT method is more robust (e.g. two different sources will not merge)
             especially in a package where the centroid will be subsequently be
             redetermined using PSF fitting.   However, it is less accurate, and introduces
             biases in the centroid histogram.   The change here is the same made in the
             IRAF DAOFIND routine (see
             http://iraf.net/article.php?story=7211&query=daofind ) [1]
            """
            
            sd = sum(temp * ywt, 0)
            
            sumgd = sum(wt * sgy * sd)
            sumd = sum(wt * sd)
            sddgdx = sum(wt * sd * dgdx)
            
            hx = (sumgd - sumgx * sumd / sumOfWt) / (sumgsqy - sumgx ** 2 / sumOfWt)
            
            # HX is the height of the best-fitting marginal Gaussian.   If this is not
            # positive then the centroid does not make sense. [1]
            if (hx <= 0):
                # Reject
                continue
            
            skylvl = (sumd - hx * sumgx) / sumOfWt
            dx = (sgdgdx - (sddgdx - sdgdx * (hx * sumgx + skylvl * sumOfWt))) / (hx * sdgdxs / sigSQ)
            
            if abs(dx) >= kernelHalfDimension:   
                # Reject
                continue
            
            xcen = ix[i] + dx    #X centroid in original array
            
            # Find Y centroid
            sd = sum(temp * xwt, 1)
            
            sumgd = sum(wt * sgx * sd)
            sumd = sum(wt * sd)
            
            sddgdy = sum(wt * sd * dgdy)
            
            hy = (sumgd - sumgy * sumd / sumOfWt) / (sumgsqx - sumgy ** 2 / sumOfWt)
            
            if (hy <= 0):
                # Reject
                continue
            
            skylvl = (sumd - hy * sumgy) / sumOfWt
            dy = (sgdgdy - (sddgdy - sdgdy * (hy * sumgy + skylvl * sumOfWt))) / (hy * sdgdys / sigSQ)
            if abs(dy) >= kernelHalfDimension:
                # Reject 
                continue
            
            ycen = iy[i] + dy    #Y centroid in original array
            
            subXYArray.append( [xcen, ycen] )
            
        et = time.time()
        if verbose:
            print 'Looping over Stars time:', ( et - st )
        
        subXYArray = averageEachCluster( subXYArray, 10 )
        xySize = len(subXYArray)
        
        
        for i in range( xySize ):
            subXYArray[i] = subXYArray[i].tolist()
            # I have no idea why the positions are slightly modified. Was done originally in
            # iqTool, perhaps for minute correcting.
            subXYArray[i][0] += 1
            subXYArray[i][1] += 1
            
            subXYArray[i][0] += yoffset
            subXYArray[i][1] += xoffset
                
            
            if writeOutFlag:
                outputLines.append( " ".join( [str(subXYArray[i][0]), str(subXYArray[i][1])] )+"\n" ) 
        
        xyArray.extend(subXYArray)
            
    oet = time.time()
    overall_time = (oet-ost-drawtime)
    if verbose:
        print 'No. of objects detected:', len(xyArray)
        print 'Overall time:', overall_time, 'seconds.'
    
    if writeOutFlag:
        outputFile = open( outfile, "w" )
        outputFile.writelines( outputLines )
        outputFile.close()
    
    if timing:
        return xyArray, overall_time
    else:
        return xyArray

#------------------------------------------------------------------------------ 
def averageEachCluster( xyArray, pixApart=10.0 ):
    """
    detSources can produce multiple centers for an object. This algorithm corrects that 
    For Example: 
    626.645599527 179.495974369
    626.652254706 179.012831637
    626.664059364 178.930738423
    626.676504143 178.804093054
    626.694643376 178.242374891
    
    This function will try to cluster these close points together, and produce a single center by
    taking the mean of the cluster. This function is based off the removeNeighbors function in iqUtil.py
    
    @param xyArray: The list of centers of found stars.
    @type xyArray: List
    
    @param pixApart: The max pixels apart for a star to be considered part of a cluster. 
    @type pixApart: Number
    
    @return: The centroids of the stars sorted by the X dimension.
    @rtype: List
    """
    newXYArray = []
    xyArray.sort()
    xyArray = array( xyArray )
    xyArrayForMean = []
    xyClusterFlag = False
    j = 0
    while j < (xyArray.shape[0]):
        i = j + 1
        while i < xyArray.shape[0]:
            diffx = xyArray[j][0] - xyArray[i][0]
            if abs(diffx) < pixApart:
                diffy = xyArray[j][1] - xyArray[i][1]
                if abs(diffy) < pixApart:
                    if not xyClusterFlag:
                        xyClusterFlag = True
                        xyArrayForMean.append(j)
                    xyArrayForMean.append(i)
                    
                i = i + 1
            else:
                break
        
        if xyClusterFlag:
            xyMean = [mean( xyArray[xyArrayForMean], axis=0 ), mean( xyArray[xyArrayForMean], axis=1 )]
            newXYArray.append( xyMean[0] )
            xyArrayForMean.reverse() # Almost equivalent to reverse, except for numpy
            for removeIndex in xyArrayForMean:
                xyArray = delete( xyArray, removeIndex, 0 )
            xyArrayForMean = []
            xyClusterFlag = False
            j = j - 1
        else:
            newXYArray.append( xyArray[j] )
        
        
        j = j + 1

    return newXYArray

#------------------------------------------------------------------------------ 
def baseHeuristic( scidata, sigma, threshold ):
    '''
    A simple heuristic for rejecting empty grids or grids with only cosmic rays.
    
    @param scidata: The gridpoint data.
    @type scidata: numpy.array
    
    @param sigma: The background value for the image.
    @type sigma: float
    
    @param threshold: The threshold used by detSources.
    @type threshold: int or float
    
    @return: True if the grid is worth rejection.
    @rtype: Boolean 
    
    '''
    stars = starCandidates(scidata, background*sigma)

    if stars.size < 20:
        return True
    
    return False

#---------------------------------------------------------------------------
def starCandidates(scidata, mean=None):
    """
    Find all pixels greater than mean.
    
    @param scidata: Science data for checking.
    @type scidata: numpy.array
    
    @return: A list of all points greater than mean.
    @rtype: numpy.array
    """
    stars = []
    
    if mean is None:
       sci_copy = scidata[:,:]
       stars = where(sci_copy > (scidata.std() + scidata.mean()))
    else:
       stars = where(scidata > mean)
    return stars[0]

#------------------------------------------------------------------------------ 
def background(scidata):
    """mask out all pixels greater than 2.5 sigma
    
    @param scidata: science data array, containing only the object to be fit
    @type scidata: numpy.array
    """
    fim = []
    stars = []
    
    fim = scidata * 1.
    stars = where(fim > (1*scidata.std() + scidata.mean()))
    fim[stars] = scidata.mean()
    #nd.display(fim, frame=3)
    
    outside = where(fim < (1*scidata.std() - scidata.mean()))
    fim[outside] = scidata.mean()

    #nd.display(scidata, frame=2)
    #nd.display(fim, frame=4)
    return fim.std()

#------------------------------------------------------------------------------ 
def draw_windows( window, dispFrame=1, label=True ):
    '''
    
    
    '''
    import pyraf
    from pyraf import iraf
    
    drawst = time.time()
    
    tmpFilename = 'tmpfile.tmp'
    index = 0
    for win in window:
        index += 1
        # The following is annoying IRAF file nonsense.
        
        tmpFile = open( tmpFilename, 'w' )
        toWrite = '%s %s W%s\n' %(str(win[0]+(win[2]/2)),str(win[1]+(win[3]/2)), str(index))
        tmpFile.write( toWrite )
        tmpFile.close()
        
        iraf.tvmark( frame=dispFrame,coords=tmpFilename, mark='rectangle',
            pointsize=8, color=204, label=label, lengths=str(win[2])+' '+str(float(win[3])/float(win[2])) )
    
    drawet = time.time()
    return drawet - drawst