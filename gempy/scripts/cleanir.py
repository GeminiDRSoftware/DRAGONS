#!/usr/bin/env python
#
# 2007 Jul 8  - Andrew W Stephens - alpha version
# 2007 Jul 9  - AWS - beta version
# 2007 Jul 10 - AWS - move most operations to cleanquad function
# 2007 Jul 11 - AWS - use stddev to decide whether to keep orig quad
# 2007 Jul 14 - AWS - generalized code to allow different pattern sizes
# 2007 Jul 18 - AWS - fix bug generating index arrays
# 2007 Jul 20 - AWS - add quadrant bias-level normalization
# 2007 Jul 23 - AWS - add force option
# 2007 Aug 06 - AWS - f/6 spectroscopy mode: use top & bottom for pattern
# 2007 Aug 22 - AWS - add -a flag to use all pixels
# 2008 Jan 11 - AWS - check for available image extensions
# 2008 Feb 05 - AWS - don't close input file until the end (req'd if next>2)
# 2008 Oct 02 - AWS - don't write pattern unless given the -p flag
# 2009 May 03 - AWS - use conformant default output file name
# 2009 May 13 - AWS - verify FITS header (old images have unquoted release date)
# 2009 May 22 - AWS - output full-frame pattern
# 2009 May 22 - AWS - improve quadrant bias normalization
# 2009 May 23 - AWS - add optional sky frame
# 2009 May 26 - AWS - add user-supplied bias offset correction
# 2009 Jul 04 - AWS - do not try to bias correct spectral flats
# 2009 Oct 24 - AWS - add basic row filtering
# 2009 Nov 06 - AWS - ignore bad pixels flagged in DQ extension
# 2009 Nov 08 - AWS - use mode for quadrant bias level normalization
# 2009 Nov 12 - AWS - use sigma-clipped stddev to judge quality of bias normalization
# 2009 Nov 17 - AWS - fit a Gaussian to the sky pixel distribution for bias norm.
# 2010 Feb 02 - AWS - sky subtract before quadrant normalization
# 2010 Feb 18 - AWS - add sigma-clipping to row filtering
# 2010 Apr 09 - AWS - only check for gcal status if OBSTYPE = FLAT
# 2010 Apr 10 - AWS - accept list input
# 2010 Apr 13 - AWS - minor tweak of the spectroscopic regions
# 2010 Jul 11 - AWS - allow images sizes which are multiples of the pattern size
# 2010 Oct 08 - AWS - option to read in bad pixel mask (e.g. object mask from nisky)
# 2010 Oct 10 - AWS - change the -g flag to take arguments
# 2010 Oct 11 - AWS - pad GNIRS images (2 row at the top)
# 2010 Oct 12 - AWS - GNIRS row filtering using an 8-pixel wide kernel
# 2010 Dec 21 - AWS - add grid filter
# 2010 Dec 28 - AWS - select GNIRS pattern region based on camera & slit
# 2011 Feb 03 - AWS - use extension 2 for nsprepared GNIRS data
# 2011 Feb 05 - AWS - add input glob expansion
# 2011 May 05 - AWS - output 32-bit files

# To Do:
# GNIRS: Mask out padding when a DQ or pixel mask is available
# Detect and mask out objects before doing any calculations
# check if run before
# properly handle images < 1024 x 1024
# Specification of image section to use to calculate pattern
# Specification of image section affected by pattern
# Look at stddev of each row to identify which have pattern noise

#-----------------------------------------------------------------------

import datetime
import getopt
import glob
import matplotlib.pyplot as pyplot
import numpy
import os
import pyfits
from scipy.optimize import leastsq
import string
import sys

version = '2011 May 5'

#-----------------------------------------------------------------------

def usage():
    print ''
    print 'NAME'
    print '       cleanir.py - filter pattern noise out of NIRI and GNIRS frames\n'
    print 'SYNOPSIS'
    print '       cleanir.py [options] infile/list\n'
    print 'DESCRIPTION'
    print '       This script assumes that the NIRI/GNIRS pattern noise in a quadrant'
    print '       can be represented by a fixed pattern which is repeated over the'
    print '       entire quadrant.  The default size for this pattern is 16 pixels'
    print '       wide and 4 pixels high (which may be changed via the -x and -y'
    print '       flags).  The pattern is determined for each quadrant by taking the'
    print '       median of the pixel value distribution at each position in the'
    print '       pattern.  Once the median pattern has been determined for a'
    print '       quadrant it is replicated to cover the entire quadrant and'
    print '       subtracted, and the mean of the pattern is added back to preserve'
    print '       flux.  The standard deviation of all the pixels in the quadrant'
    print '       is compared to that before the pattern subtraction, and if no'
    print '       reduction was achieved the subtraction is undone.  The pattern'
    print '       subtraction may be forced via the -f flag.  This process is'
    print '       repeated for all four quadrants and the cleaned frame is written'
    print '       to c<infile> (or the file specified with the -o flag).  The'
    print '       pattern derived for each quadrant may be saved with the -p flag.'
    print ''
    print '       Pattern noise is often accompanied by an offset in the bias'
    print '       values between the four quadrants.  One may want to use the'
    print '       -q flag to try to remove this offset.  This attempts to match'
    print '       the iteratively determined median value of each quadrant.'
    print '       This method works best with sky subtraction (i.e. with the -s'
    print '       flag), and does not work well if there are large extended objects'
    print '       in the frame.  By default the median is determined from the'
    print '       entire frame, although the -c flag will only use a central'
    print '       portion of the image.  Note that the derived quadrant offsets'
    print '       will be applied to the output pattern file.'
    print ''
    print '       Removing the pattern from spectroscopy is more difficult because'
    print '       of many vertical sky lines.  By default f/6 spectroscopy with the'
    print '       2-pixel or blue slits (which do not fill the detector), uses the'
    print '       empty regions at the bottom (1-272) and top (720-1024) of the'
    print '       array for measuring the pattern.  This is not possible for other'
    print '       modes of spectroscopy where the spectrum fills the detector.'
    print '       For these modes it is best to do sky subtraction before pattern'
    print '       removal.  The quickest method is to pass a sky frame (or an offset'
    print '       frame) via the -s flag.  The manual method is to generate and'
    print '       subtract the sky, determine and save the pattern via the -p flag,'
    print '       then subtract the pattern from the original image.  One may use'
    print '       the -a flag to force using all of the pixels for the pattern'
    print '       determination.'
    print ''
    print '       Note that you may use glob expansion in infile, however, the'
    print '       entire string must then be quoted or any pattern matching'
    print '       characters (*,?) must be escaped with a backslash.'
    print ''
    print 'OPTIONS'
    print '       -a : use all pixels for pattern determination'
    print '       -b <badpixelmask> : specify a bad pixel mask (overrides DQ plane)'
    print '       -c <frac> : use central <fraction> of image for bias adjustment [1]'
    print '       -d <dir> : specify an input data directory'
    print '       -f : force cleaning of all quads even if stddev does not decrease'
    print '       -g # : graph results (0=none, 1=pattern, 2=offsets, 3=both)'
    print '       -m : use median instead of fitting a Gaussian'
    print '       -o <file> : write output to <file> (instead of c<infile>)'
    print '       -p <file> : write full-frame pattern to <file>'
    print '       -q : adjust quadrant offsets'
    print '       -r : row filtering (useful for GNIRS XD spectra)'
    print '       -s <sky> : sky frame to help in pattern recognition'
    print '       -t : apply test grid filter before normalizing quadrants'
    print '       -v : verbose debugging output'
    print '       -x <size> : set pattern x size in pix [16]'
    print '       -y <size> : set pattern y size in pix [4]\n'
    print 'VERSION'
    print '       ', version
    print ''
    raise SystemExit

#-----------------------------------------------------------------------

def main():
    global allpixels, applygridfilter, bad, badmask
    global bias1, bias2, bias3, bias4, biasadjust
    global cfrac, force, graph, median, output
    global patternfile, datadir, rowfilter, savepattern
    global skyfile, skysub, subtractrowmedian, pxsize, pysize, verbose
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'ab:c:d:fg:hmo:p:qrs:tx:y:v', ['q1=','q2=','q3=','q4='])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(2)

    if (len(args)) != 1:
        usage()

    nargs = len(sys.argv[1:])
    nopts = len(opts)

    allpixels = False
    applygridfilter = False
    bad = -9.e6              # value assigned to bad pixels
    badmask = 'DQ'
    bias1 = 0.0
    bias2 = 0.0
    bias3 = 0.0
    bias4 = 0.0
    biasadjust = False
    cfrac = 1.0              # use whole image
    force = False
    graph = 0
    median = False
    output = 'default'
    patternfile = ''
    datadir = ''
    rowfilter = False
    savepattern = False
    skyfile = ''
    skysub = False
    subtractrowmedian = False
    pxsize = 16
    pysize = 4
    verbose = False

    for o, a in opts:
        if o in ('-a'):        # force using all pixels for pattern determination
            allpixels = True
        elif o in ('-b'):
            badmask = a
        elif o in ('-c'):      # use central fraction for bias normalization
            cfrac = float(a)
        elif o in ('-d'):      # input data directory
            datadir = a
        elif o in ('-f'):      # force pattern subtraction in every quadrant
            force = True
        elif o in ('-g'):      # graph results
            graph = int(a)
        elif o in ('-o'):      # specify cleaned output file
            output = a
        elif o in ('-m'):
            median = True
        elif o in ('-p'):      # write pattern file
            patternfile = a
            savepattern = True
        elif o in ('-q'):      # try to adjust quadrant bias values
            biasadjust = True
        elif o in ('--q1'):    # bias offset for quadrant 1
            bias1 = float(a)
        elif o in ('--q2'):
            bias2 = float(a)
        elif o in ('--q3'):
            bias3 = float(a)
        elif o in ('--q4'):
            bias4 = float(a)
        elif o in ('-r'):      # row filtering
            rowfilter = True
        elif o in ('-s'):      # sky frame
            skyfile = a
            skysub = True
        elif o in ('-t'):      # test grid filter
            applygridfilter = True
        elif o in ('-x'):      # specify pattern x-dimension
            pxsize = int(a)
        elif o in ('-y'):      # specify pattern y-dimension
            pysize = int(a)
        elif o in ('-v'):      # verbose debugging output
            verbose = True
        else:
            assert False, "unhandled option"

    inputfile = args[0]

    files = glob.glob(inputfile)
    if (verbose):
        print '...input = ', inputfile
        print '...files = ', files

    print ''
    
    for f in files:
        if IsFits(f):
            cleanir(f)
        else: # file list
            print 'Expanding ' + f + '...\n'
            inlist = open(f,'r')
            for line in inlist:
                cleanir(line.strip())
            inlist.close()

#-----------------------------------------------------------------------

def IsFits(infile):
    global datadir

    # If the file exists and has a .fits extension assume that it is FITS:
    if os.path.exists(datadir + infile):
        if infile.endswith('.fits'):
            fits = True
        else:
            fits = False
    elif os.path.exists(infile): # Check for lists in the CWD
        if infile.endswith('.fits'):
            fits = True
        else:
            fits = False
    else: # assume it is a FITS image missing the .fits extension
        fits = True
    return fits

#-----------------------------------------------------------------------

def IterStat(vector, lowsigma=3, highsigma=3):
    global verbose
    median = numpy.median(vector)
    stddev = numpy.std(vector)
    minval = median - lowsigma  * stddev
    maxval = median + highsigma * stddev
    num = numpy.size(vector)
    dn = 1000
    while (dn > 1 and stddev > 0 ):
        tmpvec = vector[(vector>minval) & (vector<maxval)]
        median = numpy.median(tmpvec)
        stddev = numpy.std(tmpvec)
        dn = num - numpy.size(tmpvec)
        num = numpy.size(tmpvec)
        if (verbose):
            print '   ...median=',median,' stddev=',stddev,' min=',minval,' max=',maxval,' N=',num,' dN=',dn
        minval = median - lowsigma  * stddev
        maxval = median + highsigma * stddev
    
    return (median, stddev)

#-----------------------------------------------------------------------

def CleanQuad(quad,patternin):
    # quad = quadrant to be pattern-subtracted
    # patternin = region to use for pattern determination
    global qxsize, qysize                                 # quadrant size
    global pxsize, pysize                                 # pattern size

    if (verbose):
        print '   ...mean of input quadrant =', numpy.mean(quad)
        print '   ...median of input quadrant =', numpy.median(quad)

    # create arrays of indices which correspond to the pattern tiled over
    # the region of the input quadrant to be used for pattern determination
    inpx = len(patternin[0])
    inpy = len(patternin)
    if (verbose):
        print '   ...size of pattern determination region =',inpx,'x',inpy
    indx = numpy.tile(numpy.arange(0,inpx,pxsize), inpy/pysize)
    indy = numpy.arange(0,inpy,pysize).repeat(inpx/pxsize)
    if (verbose):
        print '   ...indx:', indx
        print '   ...indy:', indy

    # create blank pattern array:
    pattern = numpy.zeros(pysize*pxsize).reshape(pysize,pxsize)
    origstddev = numpy.std(quad)
    print '   ...standard deviation of input quadrant =%9.3f' % origstddev

    # find median pattern across quadrant:
    if (graph > 0):
        binwidth = 0.5
        binmin = inputmedian - 3. * inputstddev
        binmax = inputmedian + 3. * inputstddev
        bins = numpy.arange( binmin, binmax, binwidth )
        bincenters = bins[1:bins.size] - binwidth/2.
        iplot = 0

    for iy in range(0, pysize):
        for ix in range(0, pxsize):    
            tmpdata = patternin[indy+iy,indx+ix]
            pattern[iy,ix] = numpy.median(tmpdata[tmpdata!=bad]) # filter out bad pix

            if (graph==1 or graph==3):
                iplot += 1
                hist,bins = numpy.histogram(tmpdata, bins=bins)
                plot = pyplot.subplot(pysize,pxsize,iplot)
                pyplot.plot(bincenters, hist, linestyle='', marker='.')
                pyplot.axvline(x=pattern[iy,ix], ls='--', color='green')
                if ix != 0:
                    plot.set_yticklabels([])

    if (verbose):
        print '...pattern:', pattern

    if (graph==1 or graph==3):
        print ('...graphing results...')
        pyplot.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0., hspace=0.2)
        pyplot.show()

    # tile pattern over quadrant:
    quadpattern = numpy.tile(pattern, (qysize/pysize, qxsize/pxsize))
    quadpattern -= numpy.mean(pattern)      # set the mean value to zero
    #print '   ...mean of pattern = ', numpy.mean(quadpattern)

    cleanedquad = quad - quadpattern                      # subtract pattern
    cleanstddev = numpy.std(cleanedquad)      # calculate standard deviation

    print '   ...standard deviation of cleaned quadrant = %.3f' % cleanstddev

    if (force):
        print '   ...forcing pattern subtraction'
    else:
        # has subtracting the pattern reduced the standard deviation?
        if ( origstddev - cleanstddev > 0.01 ):
            print '   ...improvement!'
        else:
            print '   ...no significant improvement so using the original quadrant'
            cleanedquad = quad      # reset quadrant pixels to original values
            quadpattern = quadpattern * 0         # set pattern to zeros

    return cleanedquad, quadpattern

#-----------------------------------------------------------------------

def CleanRow(row, sample, value):
    # row = row to be pattern-subtracted
    # sample = sample used to measure pattern
    # value = desired final median value of row

    indx = numpy.arange(0,len(sample),8)
    pattern = numpy.zeros(8)
    for ix in range(0, 8):
        tmpdata = sample[indx+ix]
        tmpdata = tmpdata[tmpdata!=bad] # filter out bad pix
        # pattern[ix] = numpy.median(tmpdata)
        median,stddev = IterStat(tmpdata)
        pattern[ix] = median

    if (verbose):
        print '...pattern:', pattern

    # repeat the pattern over the row and subtract:
    rowpattern = numpy.tile(pattern, len(row)/8)
    cleanedrow = row - rowpattern + value

    #median,stddev = IterStat(cleanedrow, lowsigma=3, highsigma=1)
    #cleanedrow = cleanedrow + (value - median)

    return cleanedrow

#-----------------------------------------------------------------------

def ApplyRowFilter(quad, patternin):
    # quad = quadrant to be pattern-subtracted
    # patternin = region to use for pattern determination
    global qxsize, qysize                                 # quadrant size

    quadmedian,quadstddev = IterStat(patternin)           # iterative median
    print '...median of input sample quadrant =', quadmedian, '+/-', quadstddev

    for iy in range(0,qysize):  # this is not correct, but will work for GNIRS
        if (verbose):
            print '...row =', iy
        quad[iy] = CleanRow(quad[iy], patternin[iy], quadmedian)
        #quad[iy] = CleanRow(quad[iy], patternin[iy], inputmedian)

    return quad

#-----------------------------------------------------------------------

def gaussian(t,p):    # p[0] = mu   p[1] = sigma    p[2] = peak
    return(p[2] * numpy.exp( -(t - p[0])**2 / (2 * p[1]**2) ))

def residuals(p,data,t):
    err = data - gaussian(t,p)
    return err

def NormQuadMedian(quad):
    global bins, bincenters, inputmean, inputmedian, inputstddev
    global lsigma, hsigma, bad
    hist,bins = numpy.histogram(quad, bins=bins)
    if (verbose):
        print '...calculating median using low-sigma =',lsigma,' and high-sigma =',hsigma
    mincts = inputmedian - lsigma * inputstddev
    maxcts = inputmedian + hsigma * inputstddev
    if (verbose):
        print '...input median=',inputmedian,' min=',mincts,' max=',maxcts
    flatquad = quad[quad != bad]  # flatten array and filter out bad pix
    npix = numpy.size(flatquad)
    dn = 100
    while (npix > 10000 and dn > 10):
        tmpquad = flatquad[(flatquad>mincts) & (flatquad<maxcts)]
        median = numpy.median(tmpquad)
        stddev = numpy.std(tmpquad)
        mincts = median - lsigma * stddev
        maxcts = median + hsigma * stddev
        dn = npix - numpy.size(tmpquad)
        npix = numpy.size(tmpquad)
        if (verbose):
            print '...median=',median,' stddev=',stddev,' min=',mincts,' max=',maxcts,' npix=',npix,' dn=',dn
    offset = inputmedian - median
    print '   ...offset = %.3f' % offset
    return hist, median, offset

#-----------------------------------------------------------------------

def NormQuadGauss(quad):
    global bins, bincenters, inputmean, inputmedian, inputstddev
    hist,bins = numpy.histogram(quad, bins=bins)
    mode = bins[ hist.argmax() ]
    peak = hist.max()
    fitsigma = 1.0      # this should probably be a command-line parameter
    mincts = mode - fitsigma * inputstddev
    maxcts = mode + fitsigma * inputstddev
    t = bincenters[ (bincenters>mincts) & (bincenters<maxcts) ]
    data =    hist[ (bincenters>mincts) & (bincenters<maxcts) ]
    p0 = [mode, inputstddev, peak]
    print '   ...initial parameter guesses = %.3f %.3f %.0f' % (p0[0],p0[1],p0[2])
    pbest = leastsq(residuals, p0, args=(data,t), full_output=1)
    p = pbest[0]
    print '   ...best fit parameters =       %.3f %.3f %.0f' % (p[0],p[1],p[2])
    offset = inputmean - p[0]
    print '   ...offset = %.3f' % offset
    xfit = numpy.linspace(mincts, maxcts, 100)
    yfit = gaussian(xfit, p)
    return hist, p[0], offset, xfit, yfit

#-----------------------------------------------------------------------

def GridFilter(img):
    global qxsize, qysize                                 # quadrant size
    gsize = 64
    indx = numpy.tile(numpy.arange(0,qxsize-gsize+1,gsize), qysize/gsize)
    indy = numpy.arange(0,qysize-gsize+1,gsize).repeat(qxsize/gsize)
    tmpimg = numpy.zeros((gsize, gsize))
    for iy in range(0, gsize):
        for ix in range(0, gsize):
            tmpdata = img[indy+iy,indx+ix]
            tmpimg[iy,ix] = numpy.median(tmpdata)
    return tmpimg

#-----------------------------------------------------------------------

def cleanir(inputfile):
    global allpixels, badmask, biasadjust, cfrac, force, median, rowfilter, skysub, verbose
    global bias1, bias2, bias3, bias4
    global inputmedian, inputstddev
    global datadir, output, pattern, patternfile, skyfile, pxsize, pysize, qxsize, qysize
    global bins, bincenters, inputmean, imputmedian, inputstddev, lsigma, hsigma

    print 'CLEANIR v.', version

    havedq = False           # we have DQ information

    if (verbose):
        print '...inputfile =', inputfile
        print '...allpixels =', allpixels
        print '...badmask =', badmask
        print '...biasadjust =', biasadjust
        print '...bias1 =', bias1
        print '...bias2 =', bias2
        print '...bias3 =', bias3
        print '...bias4 =', bias4
        print '...cfrac =', cfrac
        print '...datadir =', datadir
        print '...force =', force
        print '...median =', median
        print '...output =', output
        print '...patternfile =', patternfile
        print '...row filter =', rowfilter
        print '...skyfile =', skyfile
        print '...skysub =', skysub
        print '...pxsize =', pxsize
        print '...pysize =', pysize

    if not inputfile.endswith('.fits'):
        inputfile = inputfile + '.fits'

    if (output == 'default'):
        outputfile = 'c' + os.path.basename(inputfile)
    else:
        outputfile = output
        if ( not outputfile.endswith('.fits') ):
            outputfile = outputfile + '.fits'

    if (datadir != ''):
        inputfile = datadir + '/' + inputfile

    print 'Removing pattern noise from', inputfile

    if (savepattern):
        if ( not patternfile.endswith('.fits') ):
            patternfile = patternfile + '.fits'

    if (skysub):
        if ( not skyfile.endswith('.fits') ):
            skyfile = skyfile + '.fits'

    if not os.path.exists(inputfile):      # check whether input file exists
        print 'ERROR: ', inputfile, 'does not exist'
        sys.exit(2)

    if os.path.exists(outputfile):        # check whether output file exists
        print '...removing old', outputfile
        os.remove(outputfile)

    if (savepattern):
        if os.path.exists(patternfile):  # check whether pattern file exists
            print '...removing old', patternfile
            os.remove(patternfile)

    if (skysub):
        if not os.path.exists(skyfile):      # check whether sky file exists
            print skyfile, 'does not exist'
            sys.exit(2)

    if (cfrac < 0.1):
        print 'ERROR: central fraction must be >= 0.1'
        sys.exit(2)

    if (cfrac > 1):
        print 'ERROR: central fraction must be <= 1'
        sys.exit(2)

    if (verbose):
        print '...reading', inputfile
    hdulist = pyfits.open(inputfile)
    if (verbose):
        print '...hdulist:', hdulist.info()

    if (verbose):
        print '...verifying FITS header...'
        hdulist.verify('fix')
    else:
        hdulist.verify('silentfix')

    next = len(hdulist)
    if (verbose):
        print '...number of extensions = ', next

    if ( next == 1 ):
        sci = 0
    elif ( next < 5 ):
        sci = 1
    else:
        sci = 2
    if (verbose):
        print '...assuming the science data are in extension', sci

    image = numpy.array(hdulist[sci].data)
    if (verbose):
        print '...SCI: ', image
        print image.dtype.name

    try:
        naxis1,naxis2 = hdulist[sci].header['naxis1'], hdulist[sci].header['naxis2']
    except:
        print 'ERROR: cannot get the dimensions of extension ', sci
        pyfits.info(inputfile)
        sys.exit(2)
    print '...image dimensions = ', naxis1, 'x', naxis2

    try:
        instrument = hdulist[0].header['INSTRUME']
        if (verbose):
            print '...instrument =', instrument
    except:
        print 'WARNING: cannot determine instrument'
        instrument = 'INDEF'
        allpixels = True

    try:
        nscut = hdulist[0].header['NSCUT']
        nscut = True
    except:
        nscut = False
    if (verbose):
        print '...nscut =', nscut

    if instrument == 'GNIRS':
        print '...padding the top of GNIRS image...'
        pad = numpy.zeros((2,naxis1), dtype=numpy.float32) # create 2D array of padding        
        image = numpy.append(image,pad,axis=0)  # append the padding array to the end
        if (verbose):
            print '...new image: ', image
        naxis2 = naxis2 + 2
        print '...image dimensions = ', naxis1, 'x', naxis2

    print '...pattern size =', pxsize, 'x', pysize

    qxsize = naxis1 / 2                                       # quadrant x size
    qysize = naxis2 / 2                                       # quadrant y size

    if qxsize%pxsize != 0 or qysize%pysize != 0:
        print 'ERROR: quadrant size is not a multiple of the pattern size'
        sys.exit(2)

    if pxsize > qxsize or pysize > qysize:
        print 'ERROR: pattern size is larger than the quadrant size!'
        sys.exit(2)

    #-----------------------------------------------------------------------

    if badmask == 'DQ':
        if (verbose):
            print '...reading data quality extension...'
        try:
            dq = numpy.array(hdulist['DQ'].data)
            havedq = True
            if (verbose):
                print '...DQ: ', dq
            if ( numpy.size(dq[dq>0]) > numpy.size(dq)/2 ):
                print 'WARNING:', numpy.size(dq[dq>0]), 'pixels are flagged as bad in the DQ plane!'
        except:
            print '...no DQ data found'
            # dq = numpy.zeros(naxis2*naxis1,int)
            # dq.resize(naxis2,naxis1)
            
    elif os.path.exists(badmask): # bad pixel mask specified on the command line

        if (verbose):
            print '...reading bad pixel mask', badmask

        if badmask.endswith('.pl'):
            if (verbose):
                print '...converting pl to fits'
            # fitsfile = inputfile.replace('.pl', '.fits')
            tmpbadmask = 'cleanir-badpixelmask.fits'

            if os.path.exists(tmpbadmask):
                os.remove(tmpbadmask)

            from pyraf import iraf

            iraf.imcopy(badmask, tmpbadmask)
            badmask = tmpbadmask

        badmaskhdu = pyfits.open(badmask)

        if (verbose):
            badmaskhdu.info()

        dq = numpy.array(badmaskhdu[0].data)
        havedq = True

        if badmask.endswith('.pl'):
            os.remove(tmpbadmask)

        if (verbose):
            print '...DQ: ', dq

        badmaskhdu.close()

    else:
        print 'WARNING: ', badmask, 'does not exist'
        print 'Turning off quadrant normalization'
        biasadjust = False


    #-----------------------------------------------------------------------

    if (biasadjust):
        try:
            obstype = hdulist[0].header['OBSTYPE']
        except:
            print 'WARNING: cannot determine obstype'
            obstype = 'INDEF'
        if (verbose):
            print '...obstype =', obstype


        if (obstype == 'FLAT'):
            try:
                gcalshutter = hdulist[0].header['GCALSHUT']
            except:
                print 'WARNING: cannot determine GCAL shutter status'
                if (verbose):
                    print '...gcal shutter =', gcalshutter
            if (gcalshutter == 'OPEN'):
                print '...this is a lamps-on flat, so turning off bias normalization...'
                biasadjust = False

    # Bias level adjustment should probably only be done on flat-fielded data.

    #-----------------------------------------------------------------------

    if (skysub):
        print '...reading sky', skyfile
        sky = pyfits.open(skyfile)
        print '...verifying sky...'
        if (verbose):
            sky.verify('fix')
        else:
            sky.verify('silentfix')
        skyimage = numpy.array(sky[sci].data)

        if instrument == 'GNIRS':
            print '...padding the top of the GNIRS sky...'
            skyimage = numpy.append(skyimage,pad,axis=0)  # append the padding array to the end

    # NEED ERROR CHECKING HERE! (extensions, image size, filter, instrument, etc.)

    #-----------------------------------------------------------------------

    if (subtractrowmedian):
        print '...subtracting the median of each rows...'
        imagemean = numpy.mean(image)
        for iy in range(0, naxis2):
            tmprow = image[iy,:]
            if (verbose):
                print '...row ', iy
            median,stddev = IterStat(tmprow)  # iterative median
            image[iy,:] -= median
        image += ( imagemean - image.mean() )  # reset image to the original mean value

        # image[iy,:] -= numpy.median(image[iy,:])  # simple row-filtering over the whole image
        # Median filter each quadrant:
        # image[iy,0:naxis1/2]      -= numpy.median(image[iy,0:naxis1/2])
        # image[iy,naxis1/2:naxis1] -= numpy.median(image[iy,naxis1/2:naxis1])


    #-----------------------------------------------------------------------
    # Set regions to be used for pattern determination:
    # +-------+
    # | 1 | 2 |
    # +---+---+
    # | 3 | 4 |
    # +---+---+

    if instrument == 'NIRI':
        camera = 'INDEF'
        decker = 'INDEF'
        try:
            fpmask = hdulist[0].header['FPMASK']
        except:
            print 'WARNING: cannot find FPMASK header keyword'
            print '   Assuming that this is imaging...'
            fpmask = 'f6-cam_G5208'

    elif instrument == 'GNIRS':
        fpmask = 'INDEF'
        try:
            camera = hdulist[0].header['CAMERA']
        except:
            print 'WARNING: cannot find CAMERA header keyword'
            camera = 'INDEF'

        try:
            decker = hdulist[0].header['DECKER']
        except:
            print 'WARNING: cannot find DECKER header keyword'
            decker = 'INDEF'

    else:
        fpmask = 'INDEF'
        camera = 'INDEF'
        decker = 'INDEF'
        allpixels = True

    if (verbose):
        print '...fpmask = ', fpmask
        print '...camera = ', camera
        print '...decker = ', decker


    if allpixels:
        print '...using whole image for pattern determination'
        q1x1,q1x2, q1y1,q1y2 = 0,qxsize,      qysize,naxis2  # quad 1
        q2x1,q2x2, q2y1,q2y2 = qxsize,naxis1, qysize,naxis2  # quad 2
        q3x1,q3x2, q3y1,q3y2 = 0,qxsize,           0,qysize  # quad 3
        q4x1,q4x2, q4y1,q4y2 = qxsize,naxis1,      0,qysize  # quad 4
        lsigma = 3.0
        hsigma = 1.0      # set a very small upper threshold to reject stars

    elif   fpmask == 'f6-2pixBl_G5214' or \
           fpmask == 'f6-4pixBl_G5215' or \
           fpmask == 'f6-6pixBl_G5216' or \
           fpmask == 'f6-2pix_G5211':
        print '...using region above and below slit (y<=272 and y>=728) for pattern determination'
        q1x1,q1x2, q1y1,q1y2 = 0,qxsize,      728,naxis2
        q2x1,q2x2, q2y1,q2y2 = qxsize,naxis1, 728,naxis2
        q3x1,q3x2, q3y1,q3y2 = 0,qxsize,      0,272
        q4x1,q4x2, q4y1,q4y2 = qxsize,naxis1, 0,272
        lsigma = 3.0
        hsigma = 3.0

    elif     fpmask == 'f6-4pix_G5212'  or \
             fpmask == 'f6-6pix_G5213'  or \
             fpmask == 'f32-6pix_G5229' or \
             fpmask == 'f32-9pix_G5230':
        print '...using whole image for pattern determination'
        print 'WARNING: Sky lines may be altered by pattern removal!'
        q1x1,q1x2, q1y1,q1y2 = 0,qxsize,      qysize,naxis2
        q2x1,q2x2, q2y1,q2y2 = qxsize,naxis1, qysize,naxis2
        q3x1,q3x2, q3y1,q3y2 = 0,qxsize,           0,qysize
        q4x1,q4x2, q4y1,q4y2 = qxsize,naxis1,      0,qysize
        lsigma = 3.0
        hsigma = 3.0

    elif 'Short' in camera and decker != 'SC_XD':
        print '...using x<=160 and x>=864 for pattern determination'
        q1x1,q1x2, q1y1,q1y2 = 0,160,         qysize,naxis2
        q2x1,q2x2, q2y1,q2y2 = 864,naxis1,    qysize,naxis2
        q3x1,q3x2, q3y1,q3y2 = 0,160,         0,qysize
        q4x1,q4x2, q4y1,q4y2 = 864,naxis1,    0,qysize
        lsigma = 3.0
        hsigma = 3.0

    else:
        print '...using whole image for pattern determination'
        q1x1,q1x2, q1y1,q1y2 = 0,qxsize,      qysize,naxis2  # quad 1
        q2x1,q2x2, q2y1,q2y2 = qxsize,naxis1, qysize,naxis2  # quad 2
        q3x1,q3x2, q3y1,q3y2 = 0,qxsize,           0,qysize  # quad 3
        q4x1,q4x2, q4y1,q4y2 = qxsize,naxis1,      0,qysize  # quad 4
        lsigma = 3.0
        hsigma = 1.0      # set a very small upper threshold to reject stars

    patternin  = image.copy()
    patternin1 = patternin[q1y1:q1y2, q1x1:q1x2]
    patternin2 = patternin[q2y1:q2y2, q2x1:q2x2]
    patternin3 = patternin[q3y1:q3y2, q3x1:q3x2]
    patternin4 = patternin[q4y1:q4y2, q4x1:q4x2]

    #-------------------------------------------------------------------
    # Subtract sky frame
    if (skysub):
        print '...subtracting sky...'
        patternin1 -= skyimage[q1y1:q1y2, q1x1:q1x2]
        patternin2 -= skyimage[q2y1:q2y2, q2x1:q2x2]
        patternin3 -= skyimage[q3y1:q3y2, q3x1:q3x2]
        patternin4 -= skyimage[q4y1:q4y2, q4x1:q4x2]

    #-------------------------------------------------------------------
    # Flag pixels with bad DQ
    if (havedq):
        print '...flagging bad pixels...'
        dq1 = dq[q1y1:q1y2, q1x1:q1x2]
        dq2 = dq[q2y1:q2y2, q2x1:q2x2]
        dq3 = dq[q3y1:q3y2, q3x1:q3x2]
        dq4 = dq[q4y1:q4y2, q4x1:q4x2]
        patternin1[dq1==1] = bad
        patternin2[dq2==1] = bad
        patternin3[dq3==1] = bad
        patternin4[dq4==1] = bad

    #-------------------------------------------------------------------
    # Calculate means and medians for reference:

    inputmean = numpy.mean(image)
    print '...mean of input image = %.3f' % inputmean

    if (biasadjust or graph>1):
        #inputmedian, inputstddev = IterStat(image)  # sigma-clipped
        allpatternin = numpy.concatenate(seq=(patternin1,patternin2,patternin3,patternin4))
        allpatternin = allpatternin[allpatternin!=bad] # filter out bad values
        inputmedian, inputstddev = IterStat(allpatternin)
        print '...sigma-clipped median = %.3f' % inputmedian
        print '...sigma-clipped stddev = %.3f' % inputstddev

    #-------------------------------------------------------------------
    # calculate and subtract pattern:

    quads = image.copy()
    quad1 = quads[qysize:naxis2,      0:qxsize]
    quad2 = quads[qysize:naxis2, qxsize:naxis1]
    quad3 = quads[    0:qysize,       0:qxsize]
    quad4 = quads[    0:qysize,  qxsize:naxis1]

    print '...upper left quadrant:'
    clean1, pattern1 = CleanQuad(quad1,patternin1)
    print '...upper right quadrant:'
    clean2, pattern2 = CleanQuad(quad2,patternin2)
    print '...lower left quadrant:'
    clean3, pattern3 = CleanQuad(quad3,patternin3)
    print '...lower right quadrant:'
    clean4, pattern4 = CleanQuad(quad4,patternin4)
    
    if (verbose):
        print '...reassembling new image...'
    newimage = image.copy()
    newimage[qysize:naxis2,      0:qxsize] = clean1
    newimage[qysize:naxis2, qxsize:naxis1] = clean2
    newimage[     0:qysize,      0:qxsize] = clean3
    newimage[     0:qysize, qxsize:naxis1] = clean4

    if (verbose):
        print '...updating header...'
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    if (verbose):
        print '...time stamp =', timestamp
    hdulist[0].header.add_history('Cleaned with cleanir.py ' + timestamp)

    #-----------------------------------------------------------------------
    # Use the cleaned image from here on out
    
    patternin  = newimage.copy()

    if (skysub):
        print '...subtracting sky...'
        patternin -= skyimage

    patternin1 = patternin[q1y1:q1y2, q1x1:q1x2]
    patternin2 = patternin[q2y1:q2y2, q2x1:q2x2]
    patternin3 = patternin[q3y1:q3y2, q3x1:q3x2]
    patternin4 = patternin[q4y1:q4y2, q4x1:q4x2]

    #-----------------------------------------------------------------------
    # GNIRS 8-pixel row filtering

    # Go through each row of each quadrant and generate an 8-pixel wide kernel,
    # subtract it, and then add back the previously measured quadrant mean.

    if rowfilter:
        print '...filtering rows...'

        print '...upper left quadrant:'
        clean1 = ApplyRowFilter(newimage[qysize:naxis2,      0:qxsize], patternin1)
        print '...upper right quadrant:'
        clean2 = ApplyRowFilter(newimage[qysize:naxis2, qxsize:naxis1], patternin2)
        print '...lower left quadrant:'
        clean3 = ApplyRowFilter(newimage[     0:qysize,      0:qxsize], patternin3)
        print '...lower right quadrant:'
        clean4 = ApplyRowFilter(newimage[     0:qysize, qxsize:naxis1], patternin4)

        if (verbose):
            print '...reassembling image...'

        newimage[qysize:naxis2,      0:qxsize] = clean1
        newimage[qysize:naxis2, qxsize:naxis1] = clean2
        newimage[     0:qysize,      0:qxsize] = clean3
        newimage[     0:qysize, qxsize:naxis1] = clean4

        # Use the cleaned image from here on out
        patternin  = newimage.copy()
        if (skysub):
            print '...subtracting sky...'
            patternin -= skyimage
        patternin1 = patternin[q1y1:q1y2, q1x1:q1x2]
        patternin2 = patternin[q2y1:q2y2, q2x1:q2x2]
        patternin3 = patternin[q3y1:q3y2, q3x1:q3x2]
        patternin4 = patternin[q4y1:q4y2, q4x1:q4x2]

    #-------------------------------------------------------------------
    # Normalize each quadrant:

    if (biasadjust):
        print '...normalizing the bias level of each quadrant...'
        # And apply the measured offset to the pattern output

        if (havedq):                                # Flag pixels with bad DQ
            print '...flagging bad pixels...'
            dq1 = dq[qysize:(1+cfrac)*qysize, (1-cfrac)*qxsize:qxsize]
            dq2 = dq[qysize:(1+cfrac)*qysize, qxsize:(1+cfrac)*qxsize]
            dq3 = dq[(1-cfrac)*qysize:qysize, (1-cfrac)*qxsize:qxsize]
            dq4 = dq[(1-cfrac)*qysize:qysize, qxsize:(1+cfrac)*qxsize]
            patternin1[dq1==1] = bad
            patternin2[dq2==1] = bad
            patternin3[dq3==1] = bad
            patternin4[dq4==1] = bad

        binmin = inputmedian - 5. * inputstddev
        binmax = inputmedian + 5. * inputstddev
        binwidth = 1.0
        if (binmax - binmin) / binwidth < 50: # if too few bins the least-squares minimization will fail
            binwidth = (binmax - binmin) / 50.
        
        if (verbose):
            print '...median =', inputmedian,'  stddev =', inputstddev
        bins = numpy.arange( binmin, binmax, binwidth )
        bincenters = bins[1:bins.size] - binwidth/2.
        print '...binning into', bins.size, 'bins from', binmin, 'to', binmax

        if (applygridfilter):
            print '...applying grid filter to each quadrant...'
            patternin1 = GridFilter(patternin1)
            patternin2 = GridFilter(patternin2)
            patternin3 = GridFilter(patternin3)
            patternin4 = GridFilter(patternin4)

        if (median):
            fit = False
            print '...Using median for quadrant normalization.'
            print '...upper left quadrant:'
            hist1,center1,offset1 = NormQuadMedian(patternin1)
            print '...upper right quadrant:'
            hist2,center2,offset2 = NormQuadMedian(patternin2)
            print '...lower left quadrant:'
            hist3,center3,offset3 = NormQuadMedian(patternin3)
            print '...lower right quadrant:'
            hist4,center4,offset4 = NormQuadMedian(patternin4)
        else:
            fit = True
            print '...upper left quadrant:'
            hist1,center1,offset1,xfit1,yfit1 = NormQuadGauss(patternin1)
            print '...upper right quadrant:'
            hist2,center2,offset2,xfit2,yfit2 = NormQuadGauss(patternin2)
            print '...lower left quadrant:'
            hist3,center3,offset3,xfit3,yfit3 = NormQuadGauss(patternin3)
            print '...lower right quadrant:'
            hist4,center4,offset4,xfit4,yfit4 = NormQuadGauss(patternin4)

        newimage[qysize:naxis2,      0:qxsize] += offset1
        newimage[qysize:naxis2, qxsize:naxis1] += offset2
        newimage[0:qysize,           0:qxsize] += offset3
        newimage[0:qysize,      qxsize:naxis1] += offset4
        pattern1 -= offset1
        pattern2 -= offset2
        pattern3 -= offset3
        pattern4 -= offset4

        print '...checking quality of bias normalization...'
        newmedian, newstddev = IterStat(newimage)

        if ( inputstddev - newstddev > 0.001 ):
            print '   ...sigma-clipped stddev has decreased from %.3f to %.3f' % (inputstddev, newstddev)
            offset = inputmean - numpy.mean(newimage)
            print '...adjusting whole image by %.3f to match input image...' % offset
            newimage += offset
        else:
            print '   ...sigma-clipped stddev has not significantly improved: %.3f -> %.3f' % (inputstddev, newstddev)
            print '   ...undoing quadrant bias offsets...'
            outimage = newimage
            image[qysize:naxis2,      0:qxsize] -= offset1
            image[qysize:naxis2, qxsize:naxis1] -= offset2
            image[0:qysize,           0:qxsize] -= offset3
            image[0:qysize,      qxsize:naxis1] -= offset4
            pattern1 += offset1
            pattern2 += offset2
            pattern3 += offset3
            pattern4 += offset4

        #-------------------------------------------------------------------

        if (graph>1): # 2x2 grid of pixel distributions, fits & estimated sky values
            print ('...graphing pixel distributions in each quadrant...')

            xlimits = numpy.array([binmin, binmax])

            plot = pyplot.subplot(2,2,1)
            pyplot.plot(bincenters, hist1, linestyle='', marker='.')
            if fit:
                pyplot.plot(xfit1, yfit1, linestyle='-', color='red', linewidth=2)
            pyplot.xlim(xlimits)
            pyplot.axvline(x=center1, ls='--', color='green')
            pyplot.text(0.05, 0.85, 'mean = %.2f'  % center1, horizontalalignment='left', transform=plot.transAxes)
            pyplot.text(0.95, 0.85, 'delta = %.2f' % offset1, horizontalalignment='right', transform=plot.transAxes)
            pyplot.title('Quadrant 1')

            plot = pyplot.subplot(2,2,2)
            pyplot.plot(bincenters, hist2, linestyle='', marker='.')
            if fit:
                pyplot.plot(xfit2, yfit2, linestyle='-', color='red', linewidth=2)
            pyplot.xlim(xlimits)
            pyplot.axvline(x=center2, ls='--', color='green')
            pyplot.text(0.05, 0.85, 'mean = %.2f' % center2, horizontalalignment='left', transform=plot.transAxes)
            pyplot.text(0.95, 0.85, 'delta = %.2f' % offset2, horizontalalignment='right', transform=plot.transAxes)
            pyplot.title('Quadrant 2')
        
            plot = pyplot.subplot(2,2,3)
            pyplot.plot(bincenters, hist3, linestyle='', marker='.')
            if fit:
                pyplot.plot(xfit3, yfit3, linestyle='-', color='red', linewidth=2)
            pyplot.xlim(xlimits)
            pyplot.axvline(x=center3, ls='--', color='green')
            pyplot.text(0.05, 0.85, 'mean = %.2f' % center3, horizontalalignment='left', transform=plot.transAxes)
            pyplot.text(0.95, 0.85, 'delta = %.2f' % offset3, horizontalalignment='right', transform=plot.transAxes)
            pyplot.title('Quadrant 3')

            plot = pyplot.subplot(2,2,4)
            pyplot.plot(bincenters, hist4, linestyle='', marker='.')
            if fit:
                pyplot.plot(xfit4, yfit4, linestyle='-', color='red', linewidth=2)
            pyplot.xlim(xlimits)
            pyplot.axvline(x=center4, ls='--', color='green')
            pyplot.text(0.05, 0.85, 'mean = %.2f' % center4, horizontalalignment='left', transform=plot.transAxes)
            pyplot.text(0.95, 0.85, 'delta = %.2f' % offset4, horizontalalignment='right', transform=plot.transAxes)
            pyplot.title('Quadrant 4')
            
            pyplot.subplots_adjust(left=0.05, bottom=0.05, right=0.95,
                                   top=0.95, wspace=0.2, hspace=0.2)

            # top label = inputfile
            pyplot.show()

    #-------------------------------------------------------------------
    # Apply manual bias correction if supplied:
    if (bias1 != 0.0 or bias2 != 0.0 or bias3 != 0.0 or bias4 != 0.0):
        print '...applying user-supplied bias offset...'
        newimage[qysize:naxis2,      0:qxsize] += bias1
        newimage[qysize:naxis2, qxsize:naxis1] += bias2
        newimage[0:qysize,           0:qxsize] += bias3
        newimage[0:qysize,      qxsize:naxis1] += bias4
        pattern1 -= bias1
        pattern2 -= bias2
        pattern3 -= bias3
        pattern4 -= bias4

    if (verbose):
        print '...mean of input image = %.3f' % inputmean
        print '...mean of output image = %.3f' % numpy.mean(newimage)
        print '...median of output image = %.3f' % numpy.median(newimage)

    #-------------------------------------------------------------------
    # Write cleaned output image

    if instrument == 'GNIRS':
        print '...removing GNIRS padding...'
        # remove 2-pixel padding on top of image:
        # syntax: delete(array, [rows to delete], axis=0)
        newimage = numpy.delete(newimage, [naxis2-1,naxis2-2], axis=0)

    print '...writing', outputfile
    hdulist[sci].data = newimage
    hdulist.writeto(outputfile)

    #-------------------------------------------------------------------
    # Write pattern image

    if (savepattern):
        print '...assembling and writing pattern image...'
        # create blank pattern array:
        fullpattern = numpy.zeros(naxis2*naxis1).reshape(naxis2,naxis1)
        # assemble the quadrants into a full pattern image:
        fullpattern[qysize:naxis2,      0:qxsize] = pattern1
        fullpattern[qysize:naxis2, qxsize:naxis1] = pattern2
        fullpattern[     0:qysize,      0:qxsize] = pattern3
        fullpattern[     0:qysize, qxsize:naxis1] = pattern4
        # normalize to zero:
        fullpattern -= fullpattern.mean()
        print '...writing', patternfile
        hdu = pyfits.PrimaryHDU(fullpattern)
        hdu.writeto(patternfile)

    #-------------------------------------------------------------------
    # Close file
    
    hdulist.close()
    print ' '

#-----------------------------------------------------------------------

if __name__ == '__main__':
    main()

#-----------------------------------------------------------------------
