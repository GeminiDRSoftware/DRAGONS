

# Import core Python modules
import time


# Import scientific modules
import pyfits
import numdisplay
import numpy as np

# Import IQ modules
import iqUtil
import fit

from utils import gemutil, mefutil, filesystem

import IQTool
# This is older imports to have it working with util stuff
# in pygem.
#from iq import fit
#from iq import util
#from pygem import gemutil, mefutil, irafutil, filesystem


#---------------------------------------------------------------------------
def gemiq(image, outFile='default', function='both', verbose=True,\
          residuals=False, display=True, \
          interactive=False, rawpath='.', prefix='auto', \
          observatory='gemini-north', clip=True, \
          sigma=2.3, pymark=True, niters=4, boxSize=2., mosaic=False,
          debug=False, garbageStat=False, qa=False):
    """get psf measurements of stars

    @param image: input filename or number if using today's date
    @type image: string or int
    
    @param outFile='default': output file with fit parameters, if default uses image
                               filename+.log
    @type outFile: string
    
    @param function='both': function to fit psf. Currently supported values:
                     moffat/gauss/both
                     where both fits moffat and gaussian functions to data
    @type function: string
    
    @param verbose=True: print information to screen, includes printing psf values
    @type verbose: Boolean
    
    @param residuals=False: create and display residual subtracted images
    @type residuals: Boolean
    
    @param display=True: display images
    @type display: Boolean
    
    @param pymark=True: mark images with X's for all iraf.daofind outputs and
        circles with radius FWHM of fits
    @type pymark: Boolean
    
    @param interactive=False: if interactive = False, allow iraf.daofind to select
        star centers. otherwise, if interactive = True, allow user to select
        object centers using 'space' or 'a'
    @type interactive: Boolean
    
    @param rawpath='.': path for images if not current directory
    @type rawpath: string
    
    @param prefix='auto': if image is a number, then prefix will be of form 'N20080915'
    @type prefix: string
    
    @param observatory='gemini-north': decides whether first letter of filename is
        'N' or 'S'
    @type observatory: string
    
    @param clip=True: sigma clip measured FWHMarcsec and elliticities to remove
        outliers that snuck 
        through. Produces a mean seeing and ellipticity for enire image with standard
        deviation
    @type clip: Boolean
    
    @param sigma=2.5: if clip=True, sigma is the minimum number of standard deviations
        away from the mean that an object is clipped from the final catalogue
    @type sigma: float or int
    
    @param niters=2: integer number of times sigma clipping is iterated through
        parameter list
    @type niter: int
    
    @param boxSize=2.: aperture size around object x,y to make postage stamp for
        the fitting function
    @type boxSize: float or int
    
    @param debug=False: very verbose, print all objects removed from catalogue
        because they were close to detector edge or too close to neighbors.
    @type debug: Boolean
    """

    imagelist = []
    
    if not verbose:
        debug = False
    
    try:
        image=int(image)
        imagelist.append(image)
    except: 
        if image[0] == '@': # image is list add comma delineated --> glob list
            imagelist = open(image[1:len(image)],'r')
        else:
            imagelist.append(image)

    for image in imagelist:
        #remove the \n from end of each image name if it is there
        try:
            image=int(image)
        except:
            if image[-1]=='\n': image = image[0:len(image)-1]    

        print "IMAGE:", image
        filename, imagenorawpath = gemutil.imageName(image, rawpath, prefix=prefix, \
                                                  observatory=observatory, \
                                                  verbose=verbose)

        if outFile == 'default': outFile = imagenorawpath[0:len(imagenorawpath)-5]+'.dat'

        # Open the output parameter file
        paroutfile=open(outFile, 'w')

        paroutfile.write('\n# '+filename+'\n')
        #open image array
        hdulist = pyfits.open(filename)
        instrument = mefutil.getkey('instrume', filename)
        pixelscale = mefutil.getkey('pixscale', filename)

        ############################## GMOS specific #############################
        if instrument == 'GMOS-N' or instrument == 'GMOS-S':
            
            if instrument == 'GMOS-N': dpixelscale = 0.0727
            if instrument == 'GMOS-S': dpixelscale = 0.0737
        
            gmoskeys = ('ELEVATIO','AZIMUTH','UT','DATE','TAMBIENT','WINDSPEE',
                        'WINDDIRE','EXPTIME',
                        'FILTER1','FILTER2','FILTER3', 'CRPA','OBSTYPE','OBSCLASS',
                        'OIWFS_ST',
                        'HUMIDITY','PRESSURE','DTAZEN','PA','PIXSCALE','NCCDS',
                        'RA', 'DEC',
                        'RAOFFSET', 'DECOFFSE', 'OIARA', 'OIADEC')
            
            keylist= mefutil.getkeys(gmoskeys,filename)
            raw = (mefutil.getkey('EXTNAME',filename) == 'not found')

            gmoskeysScihdr = ('CRVAL1','CRVAL2')
            if mosaic:
                CRVAL1, CRVAL2 = mefutil.getkeys(gmoskeysScihdr,filename, extension=0)
            else:
                CRVAL1, CRVAL2 = mefutil.getkeys(gmoskeysScihdr,filename, extension=2)
            #automatically generate this gmoskeydict with a loop
            gmoskeydict = {'ELEVATIO':keylist[0],'AZIMUTH':keylist[1],'UT':keylist[2],
                           'DATE':keylist[3], 'TAMBIENT':keylist[4],
                           'WINDSPEE':keylist[5],'WINDDIRE':keylist[6],
                           'EXPTIME':keylist[7],'FILTER1':keylist[8],'FILTER2':keylist[9],
                           'FILTER3':keylist[10], 'CRPA':keylist[11],'OBSTYPE':keylist[12],
                           'OBSCLASS':keylist[13],'OIWFS_ST':keylist[14],
                           'HUMIDITY':keylist[15],
                           'PRESSURE':keylist[16],'DTAZEN':keylist[17],'PA':keylist[18],
                           'PIXSCALE':keylist[19],'NCCDS':keylist[20],
                           'RA':keylist[21], 'DEC':keylist[22],
                           'RAOFFSET':keylist[23], 'DECOFFSE':keylist[24],
                           'OIARA':keylist[25],
                           'OIADEC':keylist[26], 'CRVAL1':CRVAL1, 'CRVAL2':CRVAL2}


            for key in gmoskeydict:
                paroutfile.write('# '+key+' = '+str(gmoskeydict[key])+'\n')

            if gmoskeydict['NCCDS'] == 'not found': NCCDS=3
            else: NCCDS=gmoskeydict['NCCDS']
            
            n=1
            allgfit = []
            allmfit = []
            # Loop through GMOS CCDS
            while n <= NCCDS:
                if n==2 and mosaic:
                    break
                scidata = hdulist[n].data # don't open twice
                scihdr = hdulist[n].header
                ccdsum = scihdr['CCDSUM']
                ccdint = int(ccdsum[0])
                if pixelscale == 'not found':
                    if ccdsum != 'not found':
                        ccdint = int(ccdsum[0])
                        pixelscale = dpixelscale * ccdint
                    else:
                        ccdint = 2
                        pixelscale = dpixelscale * ccdint
                        print 'GETIQ - WARNING: Could not find PIXELSCALE \
                            in PHU or CCDSUM in science header,\
                            Using pixelscale '+pixelscale 

                if n==1:
                    if ccdint == 1: xmin =890; xmax=2090
                    else: xmin = 445; xmax = 1045
                if n==2:
                    imageSigma = 'default'
                    xmin = None
                    xmax = None
                if n==3:
                    if ccdint == 1: xmin = 31; xmax = 1200
                    else: xmin = 31; xmax=600

                # goes in interactive
                imageSigma = iqUtil.starMask(scidata[:,xmin:xmax]).std()
                if debug: print ('\n# GETIQ - GMOS CCD: '+str(n))

                if interactive: 
                    pyexam(scidata, function, pixelscale, frame=n,
                                     outFile=paroutfile, verbose=verbose,
                                     pymark=pymark, residuals=residuals,
                                     clip=clip, debug=debug, niters=niters,
                                     boxSize=boxSize)
                else:
                    if raw:
                        gAllstars, mAllstars = pyiq(filename+'['+str(n)+']',
                            scidata, function, paroutfile, pixelscale,
                            frame=n, pverbose=verbose, pymark=pymark,
                            residuals=residuals, clip=False, sigma=sigma,
                            niters=niters, display=display,
                            imageSigma=imageSigma, boxSize=boxSize,
                            debug=debug, xmin=xmin,xmax=xmax, qa=qa)
                    else:
                        gAllstars,mAllstars = pyiq(filename+'[sci,'+str(n)+']',
                            scidata, function, paroutfile, pixelscale,
                            frame=n, pverbose=verbose, pymark=pymark,
                            residuals=residuals, clip=False, sigma=sigma,
                            niters=niters, display=display,
                            imageSigma=imageSigma, boxSize=boxSize,
                            debug=debug, xmin=xmin, xmax=xmax, qa=qa)
                 
                    if gAllstars: allgfit.append(gAllstars)
                    if mAllstars: allmfit.append(mAllstars)
                n+=1
            iqdata = []
            if clip: # Clip values from all three CCDS 
                if verbose: print '\n'
            
                if allgfit:
                    allgfit2 = []
                    # allgfit is a tuple of three tuples each containing
                    # a tuple of dictionaries       
                    for ccd in allgfit:
                        for star in ccd:
                            allgfit2.append(star)
                        
    	            if verbose: print '# PYEXAM - performing Gaussian \
                            clipping of all GMOS ccds'
                            
                    allgfit2, gEllMean, gEllSigma, gFWHMMean, gFWHMSigma = \
    	        	   iqUtil.sigmaClip(allgfit2, paroutfile,
                                                    sigma, verbose, niters, garbageStat=garbageStat)
                    for star in allgfit2: #only write the clipped objects 
                        iqUtil.writePars(star, paroutfile, 'gaussian')
                    
                    g1 = '# PYEXAM - All CCDS Total Gauss Ellipticity:\
                    '+str(gEllMean)+' +/- '+str(gEllSigma)
                    g2 = '# PYEXAM - ALL CCDS Total Gauss FWHM (arcsec):\
                    '+str(gFWHMMean)+' +/- '+str(gFWHMSigma)+'\n'
                    
                    if verbose:
                        print g1+'\n'+g2
                    paroutfile.write(g1+'\n'+g2)
                    iqdata.append( (gEllMean,gEllSigma,gFWHMMean,gFWHMSigma) )
                if allmfit:
                    allmfit2 = []
                    # allmfit is a tuple of three tuples each containing
                    # a tuple of dictionaries       
                    for ccd in allmfit:
                        for star in ccd:
                            allmfit2.append(star)
    	            if verbose: print '# PYEXAM - performing Moffat clipping of all GMOS ccds'
                    
                    allmfit2, mEllMean, mEllSigma, mFWHMMean, mFWHMSigma = \
                    iqUtil.sigmaClip(allmfit2, paroutfile, sigma,
                                             debug, niters, garbageStat=garbageStat)
                    for star in allmfit2: #only write the clipped objects
                         # to the catalogue
                         iqUtil.writePars(star, paroutfile, 'moffat')
                                     
                    m1 = '# PYEXAM - All CCDS Total Moffat Ellipticity:\
                     '+str(mEllMean)+' +/- '+str(mEllSigma)
                    m2 = '# PYEXAM - ALL CCDS Total Moffat FWHM (arcsec):\
                     '+str(mFWHMMean)+' +/- '+str(mFWHMSigma)+'\n'
                    if verbose:
                        print m1+'\n'+m2
                    paroutfile.write(m1+'\n'+m2)
                    iqdata.append( (mEllMean,mEllSigma,mFWHMMean,mFWHMSigma) )
            else:
                if allgfit:
                    for ccd in allgfit:
                        for star in ccd:
                            iqUtil.writePars(star, paroutfile, 'gaussian')
                if allmfit:
                    for ccd in allmfit:
                        for star in ccd:
                            iqUtil.writePars(star, paroutfile, 'moffat')

    ############################# NIRI SPECIFIC #############################
                            
        elif instrument == 'NIRI':
            #this is where niri stuff will go
            print ' '
            print 'filename=',filename
            scidata = hdulist[1].data
            gAllstars, mAllstars = pyiq(filename+'[1]', scidata,
                function, paroutfile, pixelscale, frame=n, pverbose=verbose,
                pymark=pymark, residuals=residuals, clip=clip, sigma=sigma,
                niters=niters, display=display, imageSigma=imageSigma,
                boxSize=boxSize,debug=debug, xmin=xmin, xmax=xmax, qa=qa)

        else:
            #this is where other instrument stuff will go
            print ' '
            print 'filename=',filename
            pixelscale=1.
            imageSigma='default'
            scidata = hdulist[1].data
            gAllstars, mAllstars = pyiq(filename+'[1]', scidata,
                function, paroutfile, pixelscale, frame=1, pverbose=verbose,
                pymark=pymark, residuals=residuals, clip=clip, sigma=sigma,
                niters=niters, display=display, imageSigma=imageSigma,
                boxSize=boxSize,debug=debug, qa=qa)

        paroutfile.close()
        return iqdata
#---------------------------------------------------------------------------


def pyexam(scidata, function='both', pixelscale=1, frame=1, \
           outFile='testout.txt', verbose=True, pymark=True, \
           residuals=False, clip=True, sigma=3, boxSize=9., \
           debug=False, niters=4.):
        import pylab

        from IQTool.gemplotlib import overlay
        
        # Get box size around each object
        apertureSize = fit.getApSize(pixelscale, boxSize)
 
        outstring = '%10s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s\n'%('function', 'Cx', 'Cy', 'Bg', 'Peak', 'Wx', 'Wy', 'CooX', 'CooY', 'Ellipticity', 'FWHM_pix','FWHM_arcsec', 'Theta', 'PA_deg', ' FWHMx', 'FWHMy', 'Beta')

        outFile.write(outstring)

        print 'PYEXAM - Please select objects using spacebar, and press \'q\' when finished'
        print '         pressing k, j, r, or  e will make plots along with fits'
        keystroke = ''
        numdisplay.display(scidata, z1=scidata.mean()-scidata.std(),
                           z2=scidata.mean()+scidata.std(), frame=frame)     
        if verbose:
            print '%10s%12s%12s%12s%12s%12s%12s'%('function', 'Coox', 'CooY', 'FWHMpix',
                                                  'FWHMarcsec', 'Ellipticity', 'PAdeg')
      
        gAllstars = []
        mAllstars = []
        done = 'no'
        while done == 'no':
            # Get x,y
            
            cursorInput = numdisplay.readcursor(sample=0)
            components = cursorInput.split()
            xPos = float(components[0])
            yPos = float(components[1]) 
            keystroke = components[3]
            option = ''
            if keystroke == '\\040' or keystroke == 'a' or keystroke == 'k' or keystroke == 'j' or keystroke == 'e' or keystroke == 'r':
                print 'PYEXAM - X:',xPos,'Y:',yPos
                gfitArray=None
                mfitArray=None
                gfxn=None
                mfxn=None
                
                positionCoords = (xPos,yPos)

                stampArray, stampPars = fit.makeStamp(scidata, positionCoords,
                                          apertureSize, outFile, debug=debug)
                
                if stampArray == None: return None, None

                gfitPars, gReturnModel, mfitPars, mReturnModel = \
                    fit.fitprofile(stampArray, function, positionCoords, \
                    outFile, pixelscale, stampPars, debug=debug, frame=frame)

                imageDim = list(scidata.shape) # Gives [y,x] dimensions
                #imageDim.reverse() #give [x,y] dimensions
                
                if gfitPars != None:
                    clipobsPars = ('gauss', gfitPars['CooX'], gfitPars['CooY'],
                                   gfitPars['FWHMpix'],
                                   gfitPars['FWHMarcsec'], gfitPars['Ellip'],
                                   gfitPars['PAdeg'])
                    iqUtil.printPars(clipobsPars, verbose)
                    gAllstars.append(gfitPars)

                    if pymark:
                        overlay.circle(x=gfitPars['CooX']-1,y=gfitPars['CooY']-1, \
                                       radius=gfitPars['FWHMpix'],frame=frame,\
                                       color=204)

                    gfxn = gReturnModel((gfitPars['Bg'], gfitPars['Peak'], \
                                           gfitPars['Cy'], \
                                           gfitPars['Cx'], gfitPars['Wy'], \
                                           gfitPars['Wx'], gfitPars['Theta']+90))

                    gfitArray = np.zeros(imageDim)
                    gfitArray[0:stampPars[3],0:stampPars[1]] = gfxn(*np.indices((stampPars[3], stampPars[1])))
                    
                 
                if mfitPars != None:
                    clipobsPars = ('moffat', mfitPars['CooX'],mfitPars['CooY'],
                                   mfitPars['FWHMpix'],
                                   mfitPars['FWHMarcsec'], mfitPars['Ellip'],
                                   mfitPars['PAdeg'])
                      
                    iqUtil.printPars(clipobsPars, verbose)
                    print mfitPars['CooX']
                    print mfitPars['CooY']
                    print mfitPars['Cx']
                    print mfitPars['Cy']

                    
                    mAllstars.append(mfitPars)
                    if pymark:
                        overlay.circle(x=mfitPars['CooX'],
                                       y=mfitPars['CooY'], frame=frame,
                                       radius=mfitPars['FWHMpix'], color=212)

                    mfxn = mReturnModel((mfitPars['Bg'], mfitPars['Peak'], \
                                        mfitPars['Cy'],\
                                        mfitPars['Cx'], mfitPars['Wy'], \
                                        mfitPars['Wx'], mfitPars['Theta']+90, \
                                        mfitPars['Beta']))
                    mfitArray = np.zeros(imageDim)
                    mfitArray[0:stampPars[3],0:stampPars[1]] = mfxn(*np.indices((stampPars[3], stampPars[1])))

                                   
   
                if keystroke == 'k':
                    print "PYEXAM - Yslice Plotting"
                    pylab.clf()
                    pylab.plot(stampArray.max(0), 'ro')
                    if mfitArray != None:
                        pylab.plot(mfitArray.max(0), 'b', label='moffat fit FWHM='+str(mfitPars['FWHMarcsec']))
                        pylab.axis([0,stampPars[1]-stampPars[0],mfitPars['Bg'],1.3*(mfitPars['Bg']+mfitPars['Peak'])])
                    if gfitArray != None:
                        pylab.plot(gfitArray.max(0), 'g',label='gauss fit FWHM='+str(gfitPars['FWHMarcsec']))
                        pylab.axis([0,stampPars[1]-stampPars[0],gfitPars['Bg'],1.3*(gfitPars['Bg']+gfitPars['Peak'])])
                    pylab.legend()

                if keystroke == 'j':
                    print "PYEXAM - Xslice Plotting"
                    pylab.clf()
                    pylab.plot(stampArray.max(1), 'ro')
                    if mfitArray != None:
                        pylab.plot(mfitArray.max(1), 'b', label='moffat fit FWHM='+str(mfitPars['FWHMarcsec']))
                        pylab.axis([0,stampPars[1]-stampPars[0],mfitPars['Bg'],1.3*(mfitPars['Bg']+mfitPars['Peak'])])
                    if gfitArray != None:
                        pylab.plot(gfitArray.max(1), 'g',label='gauss fit FWHM='+str(gfitPars['FWHMarcsec']))
                        pylab.axis([0,stampPars[1]-stampPars[0],gfitPars['Bg'],1.3*(gfitPars['Bg']+gfitPars['Peak'])])
                    pylab.legend()

                if keystroke == 'r':
                    print "PYEXAM - radial Plotting doesnt work yet!"
                    print "new version"
                    #rad1 = np.sqrt(stampArray.max(1)**2 + stampArray.max(0)**2)
                    pylab.clf()
                    pylab.plot(stampArray.max(1), 'ro')
                    if mfitArray != None:
                        pylab.plot(mfitArray.max(1), 'b', label='moffat fit FWHM='+str(mfitPars['FWHMarcsec']))
                        pylab.axis([0,stampPars[1]-stampPars[0],mfitPars['Bg'],1.3*(mfitPars['Bg']+mfitPars['Peak'])])
                    if gfitArray != None:
                        pylab.plot(gfitArray.max(1), 'g',label='gauss fit FWHM='+str(gfitPars['FWHMarcsec']))
                        pylab.axis([0,stampPars[1]-stampPars[0],gfitPars['Bg'],1.3*(gfitPars['Bg']+gfitPars['Peak'])])
                    pylab.legend()
                    
                    

                if keystroke == 'e':
                    print 'PYEXAM - Plotting contours'
                    pylab.clf()
                    pylab.contour(stampArray)
                                
            elif keystroke == 'q':
                done = 'yes'
                if verbose: print "Done looping through stars"

        if clip:
            if gAllstars:
                if verbose: print "# PYEXAM - performing Gaussian clipping"

                gAllstars, gEllMean, gEllSigma, gFWHMMean, gFWHMSigma = \
                   iqUtil.sigmaClip(gAllstars, outFile, sigma, verbose, niters, garbageStat=garbageStat)
            
                if verbose:
                    print "PYEXAM - Mean Gaussian Ellipticity:", gEllMean
                    print "PYEXAM - Mean Gaussian FWHM (""):", gFWHMMean
                    
            if mAllstars:
                if verbose: print "# PYEXAM - performing Moffat clipping"

                mAllstars, mEllMean, mEllSigma, mFWHMMean, mFWHMSigma = \
                   iqUtil.sigmaClip(mAllstars, outFile, sigma, verbose, niters, garbageStat=garbageStat)

                if verbose:
                    print "PYEXAM - Mean Moffat Ellipticity:", mEllMean
                    print "PYEXAM - Mean Moffat FWHM (""):", mFWHMMean
         

        for star in gAllstars:
            iqUtil.writePars(star, outFile, 'gaussian')

        for star in mAllstars:
            iqUtil.writePars(star, outFile, 'moffat')

 #---------------------------------------------------------------------------

def pyiq (filename, scidata, function, outFile, pixelscale, \
              frame=1, pverbose=True, pymark=True, clip=True, \
              sigma=2.3, niters=4, display=True, imageSigma='default', \
              boxSize=9., residuals=False, saturation=65000,\
              debug=False, xmin=None, xmax=None, qa=False):
    '''finds stars and fits gaussian/moffat/both to them

    @param filename: input filename with ".fits[sci,i]" attached
    @type filename: string

    @param scidata: science data array, equal to pyfits.getdata(filename)
    @type scidata: numpy array

    @param function: currently supported are "gauss", "moffat", or "both"
    @type function: string

    @param outFile: opened file object
    @type outFile: string        
    
    @param pixelscale: instrument pixelscale in arcsec/pix
    @type pixelscale: float or int

    @param frame: frame number for marking, only used for multiple science extensions
    @type frame: int

    @param pverbose: warnings will be printed
    @type pverbose: Boolean

    @param pymark: mark displayed images with daofind centers and fwhm circles for fits
    @type pymark: Boolean

    @param clip: Sigma clip outliers in measured FWHM and ellipticities
    @type clip: Boolean

    @param sigma: Sigma threshold for clipping outlier FWHM and ellipticities
    @type clip: float or int

    @param niters: number of times sigma clipping is iterated through parameter list
    @type niters: int

    @param display: Display images?
    @type display: Boolean

    @param imageSigma: sigma of background level, if "default" a star mask is made
                       and std is measured
                       directly from masked image
    @type imageSigma: string

    @param boxSize: aperture size around object x,y to make postage stamp for the
                       fitting function
    @type boxSize: float

    @param residuals:this is problem
    @type residuals: Boolean

    @param saturation: saturation values for daofind inputs
    @type saturation: float or int
        '''
    
    if display!=True: pymark=False

    #initialize arrays
    if residuals:
        gsubtracted = scidata[:]
        msubtracted = scidata[:]
 
    gAllstars = []
    mAllstars = []
    xyArray = []

    #Find Objects
    xyArray = iqUtil.pyDaoFind(filename, scidata, pixelscale, \
              frame=frame, debug=debug, pymark=pymark, \
              display=display, imageSigma=imageSigma, saturation=saturation, qa=qa)
    
    # Get box size around each object
    apertureSize = fit.getApSize(pixelscale, boxSize)
    
    # remove objects that are too close to each other
    xyArray = iqUtil.removeNeighbors(xyArray, npixapart=apertureSize,
                                   crowded=True, debug=debug)
   
    # remove objects that are too close to the edge
    xyArray = iqUtil.edgeCheck(scidata,xyArray,npixaway=apertureSize-1,
                             debug=debug, xmin=xmin, xmax=xmax)

    # label printed value columns
    if debug:
        print '%10s%12s%12s%12s%12s%12s%12s'%('# function', 'Coox',
                'CooY', 'FWHMpix', 'FWHMarcsec', 'Ellip', 'PAdeg')

     # label output file columns
    if frame==1:
        outstring = '%10s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s%5s\n'\
        %('# function', 'Cx', 'Cy', 'Bg', 'Peak', 'Wx', 'Wy', 'CooX', 'CooY', \
        'Ellipticity', 'FWHM_pix', 'FWHM_arcsec', 'Theta', 'PA_deg', 'FWHMx', \
         'FWHMy', 'Beta', 'CCD')
        outFile.write(outstring)
        
    # loop through objects and fit functions 
    for [xCoo,yCoo] in xyArray:
        positionCoords= [xCoo,yCoo]
        stampArray, stampPars = fit.makeStamp(scidata, positionCoords, \
                                          apertureSize, outFile, debug=debug)
        
        st = time.time()
        gfitPars, gReturnModel, mfitPars, mReturnModel = \
            fit.fitprofile(stampArray, function, positionCoords, \
            outFile, pixelscale, stampPars, debug=debug, frame=frame)
        et = time.time()
       #print 'Fit Time', (et-st)

        ######## Gaussian Fits #########
        if gfitPars != None:
            clipobsPars = ('gauss', gfitPars['CooX'], gfitPars['CooY'],
                           gfitPars['FWHMpix'], gfitPars['FWHMarcsec'],
                           gfitPars['Ellip'], gfitPars['PAdeg'])
            iqUtil.printPars(clipobsPars, debug)
	    gAllstars.append(gfitPars)
        
            if pymark:
                iqUtil.iqmark (frame, gfitPars, color=208)

            if residuals:
                if gAllstars[0] == gfitPars:
                    gsubtracted = iqUtil.makeResiduals(scidata, gfitPars, \
                                            gReturnModel, stampPars)
                else:
		    gsubtracted = iqUtil.makeResiduals(gsubtracted, gfitPars, \
                                            gReturnModel, stampPars)


        #########  Moffat Fits ######### 
        if mfitPars != None:
            clipobsPars = ('moffat', mfitPars['CooX'], mfitPars['CooY'],
                               mfitPars['FWHMpix'], mfitPars['FWHMarcsec'],
                               mfitPars['Ellip'], mfitPars['PAdeg'])
            iqUtil.printPars(clipobsPars, debug)
            mAllstars.append(mfitPars)
                        
            if pymark:
                iqUtil.iqmark (frame, mfitPars, color=204)

            if residuals:
                if mAllstars[0] == mfitPars:
                    # subtract first object
                    msubtracted = iqUtil.makeResiduals(scidata, mfitPars, \
                                               mReturnModel, stampPars)
                else:
                    # residual subtract already subtracted image
                    msubtracted = iqUtil.makeResiduals(msubtracted, mfitPars, \
                                                mReturnModel, stampPars)
            

    if debug: print '# PYEXAM - done looping through objects'

    # write the final residual subtracted image to a fits file
    if residuals:
        resName = gemUtil.removeExtension(outFile.name)
        
        if gAllstars != []:
            filesystem.deleteFile(resName+'gaussResidual'+str(frame)+'.fits')
            pyfits.writeto(resName+'gaussResidual'+str(frame)+'.fits',\
                gsubtracted)
        if mAllstars != []:
            filesystem.deleteFile(resName+'moffatResidual'+str(frame)+'.fits')
            pyfits.writeto(resName+'moffatResidual'+str(frame)+'.fits',\
                msubtracted)       

    if clip: #this doesnt get called for GMOS as gemiq does the clipping
             # should be a separate routine in iqUtil as it is used in gemiq
        if gAllstars:
            if debug: print "# PYEXAM - performing Gaussian clipping"
            gAllstars, gEllMean, gEllSigma, gFWHMMean, gFWHMSigma = \
               iqUtil.sigmaClip(gAllstars, outFile, sigma, pverbose, niters, garbageStat)
            for star in gAllstars:
                writePars(star, outFile, 'gaussian')
                clipobsPars = ('gauss', star['CooX'], star['CooY'],
                               star['FWHMpix'], star['FWHMarcsec'],
                               star['Ellip'], star['PAdeg'])
                iqUtil.printPars(clipobsPars, pverbose)

            g1 = "# PYEXAM - Gauss Ellipticity: "+str(gEllMean)+" +/- "+str(gEllSigma)
            g2 = "# PYEXAM - Gauss FWHM (arcsec): "+str(gFWHMMean)+" +/- "+str(gFWHMSigma)+'\n'
            if pverbose:
                print g1+'\n'+g2
            outFile.write(g1+'\n'+g2)
      
        if mAllstars:
            if debug: print "# PYEXAM - performing Moffat clipping"
            mAllstars, mEllMean, mEllSigma, mFWHMMean, mFWHMSigma = \
                iqUtil.sigmaClip(mAllstars, outFile, sigma, pverbose, niters, garbageStat=garbageStat)
            for star in mAllstars:
                writePars(star, outFile, 'moffat')
                clipobsPars = ('moffat', star['CooX'], star['CooY'],
                               star['FWHMpix'], star['FWHMarcsec'],
                               star['Ellip'], star['PAdeg'])
                iqUtil.printPars(clipobsPars, pverbose)

            m1 = "# PYEXAM - Moffat Ellipticity: "+str(mEllMean)+" +/- "+str(mEllSigma)
            m2 = "# PYEXAM - Moffat FWHM (arcsec): "+str(mFWHMMean)+" +/- "+str(mFWHMSigma)+'\n'
            if pverbose:
                print m1+'\n'+m2
            outFile.write(m1+'\n'+m2)
    if debug:
        print "IQ720:", len(gAllstars), "/", len(mAllstars)
    return gAllstars, mAllstars

 #---------------------------------------------------------------------------






