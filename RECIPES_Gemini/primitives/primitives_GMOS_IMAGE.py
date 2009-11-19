#from Reductionobjects import Reductionobject
from primitives_GEMINI import GEMINIPrimitives
# All GEMINI IRAF task wrappers.
import time
from pyraf.iraf import tables, stsdas, images
from pyraf.iraf import gemini
import pyraf


import pyfits
import numdisplay
import string

yes = pyraf.iraf.yes
no = pyraf.iraf.no

# NOTE, the sys.stdout stuff is to shut up gemini and gmos startup... some primitives
# don't execute pyraf code and so do not need to print this interactive 
# package init display (it shows something akin to the dir(gmos)
import sys, StringIO, os
SAVEOUT = sys.stdout
capture = StringIO.StringIO()
sys.stdout = capture
gemini()
gemini.gmos()
sys.stdout = SAVEOUT

class GMOS_IMAGEPrimitives(GEMINIPrimitives):
    def init(self, co):
        if "global" in co and "adata" in co["global"]:
            pyraf.iraf.set (adata=co["global"]['adata'].value)  
        else:
            (root, name) = os.path.split(co.inputs[0])
            pyraf.iraf.set (adata=root)  
        
        GEMINIPrimitives.init(self, co)
        return co

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def flatCreate(self, co):

        # FLAT made with giflat
        try:
            print 'combining and normalizing best 20 twilight flats'
            gemini.giflat(co.inputsAsStr(), outflat=co["outflat"],
                bias=co.calName("REDUCED_BIAS"),rawpath=co["caldir"],
                fl_over=co["fl_over"], fl_trim=co["fl_trim"], 
                fl_vardq=co["fl_vardq"],Stdout = co.getIrafStdout(), Stderr = co.getIrafStderr())
        except:
            print "problem combining imaging flats with giflat"
            raise
        yield co

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def biasCreate(self, co):
        # Things done to the bias image before we subtract it:
        # overscan subtract
        # overscan trim
        # average combine images

        # BIAS made for all GMOS modes (imaging, spectroscopy, IFU) we need to
        # consider a generic task. using gbias (IRAF generic task)
        try:
            print "combining biases to create master bias"
            gemini.gbias(co.inputsAsStr(), outbias=co["outbias"],
                rawpath=co["caldir"], fl_trim=co["fl_trim"], 
                fl_over=co["fl_over"], fl_vardq=co["fl_vardq"],
                Stdout = co.getIrafStdout(), Stderr = co.getIrafStderr())
        except:
            print "problem combining biases with gbias"
            raise SystemExit

        yield co 

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def prepare(self, co):
        try:
            print 'preparing'
            print "Updating keywords PIXSCALE, NEXTEND, OBSMODE, GEM-TLM, GPREPARE"
            print "Updating GAIN keyword by calling GGAIN"
            gemini.gmos.gprepare(co.inputsAsStr(strippath = True), rawpath="adata$",Stdout = co.getIrafStdout(), Stderr = co.getIrafStderr())
            co.reportOutput(co.prependNames("g", currentDir = True))
            
        except:
            print "Problem in GPREPARE"
            raise 
        yield co 

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def overscanSub(self, co):
        try:
            print "Determining overscan subtraction region using nbiascontam"
            print "parameter and BIASSEC header keyword"
            print "Subtracting overscan bias levels using colbias"
            gemini.gmos.gireduce(co.inputsAsStr(strippath=True), fl_over=pyraf.iraf.yes,fl_trim=no, fl_bias=no, \
                fl_flat=no, outpref="oversub_",Stdout = co.getIrafStdout(), Stderr = co.getIrafStderr())
            co.reportOutput(co.prependNames("oversub_", currentDir = True))
        except:
            print "Problem subtracting overscan bias"
            print "Problem in GIREDUCE"
            raise
        yield co

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def overscanTrim(self, co):
        try:
            print "Determining overscan region using BIASSEC header keyword"
            print "Trimming off overscan"
            gemini.gmos.gireduce(co.inputsAsStr(), fl_over=no,fl_trim=yes, 
                fl_bias=no, fl_flat=no, outpref="trim_",Stdout = co.getIrafStdout(),
                Stderr = co.getIrafStderr())
            co.reportOutput(co.prependNames("trim_"))
        except:
            print "Problem trimming off overscan region"
            print "Problem in GIREDUCE"
            raise 
        yield co

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def overscanCorrect(self, co):
        print "Performing Overscan Correct (overSub, overTrim)"
        try:
            gemini.gmos.gireduce(co.inputsAsStr(strippath=True), fl_over=pyraf.iraf.yes,fl_trim=pyraf.iraf.yes,
                fl_bias=no, fl_flat=no, outpref="trim_oversub_",
                Stdout = co.getIrafStdout(), Stderr = co.getIrafStderr())
            co.reportOutput(co.prependNames("trim_oversub_", currentDir = True))
        except:
            print "Problem correcting overscan region"
            raise
        
        yield co
    
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def biasCorrect(self, co):
        # not really sure we need to use gireduce here. I think we could easily make a
        # more generic bias sub task
        try:
            print "Subtracting off bias"
            cals = co.calFilename( 'bias' )
            for cal in cals:
                gemini.gmos.gireduce(",".join(cals[cal]), fl_over=no,
                    fl_trim=no, fl_bias=yes,bias=cal,
                    fl_flat=no, outpref="biassub_",
                    Stdout = co.getIrafStdout(), Stderr = co.getIrafStderr()
                    ) # this flag was removed?,fl_mult=no)
            
            co.reportOutput(co.prependNames("biassub_"))
        except:
            print "Problem subtracting bias"
            print "Problem in GIREDUCE"
            raise
            
        yield co
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def flatfieldCorrect(self, co):
        try:
            print "Flat field correcting"
            
            cals = co.calFilename( 'twilight' )
            for cal in cals:
                gemini.gmos.gireduce(",".join(cals[cal]), fl_over=no,fl_trim=no,
                    fl_bias=no, flat1=cal, fl_flat=yes, outpref="flatdiv_",
                    Stdout = co.getIrafStdout(), Stderr = co.getIrafStderr()) 
            co.reportOutput(co.prependNames("flatdiv_"))   
        except:
            print "Problem dividing by normalized flat"
            print "Problem in GIREDUCE"
            raise
        yield co

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def fringeCreate(self, co):
        try:
            print "creating fringe frame"
            gemini.gifringe(co.inputsAsStr(), "fringe",
                Stdout = co.getIrafStdout(), Stderr = co.getIrafStderr())
        except:
            print "Problem creating fringe from "+co.inputsAsStr()
            print "Problem in GIFRINGE"
            raise
        yield co
        
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def fringeSubtract(self, co):
        try:
            print "subtracting fringe frame"
            gemini.girmfringe(co.inputsAsStr(), co["fringe"],
            Stdout = co.getIrafStdout(), Stderr = co.getIrafStderr())
        except:
            print "Problem subtracting fringe from "+co.inputsAsStr()
            print "Problem in GIRMFRINGE"
            raise
        yield co

    def shift(self, co):
        '''
        !!!NOTE!!!
        The code in this method was designed only for demo use. It should not be taken seriously.
        '''
        try:
            print 'shifting image'
            compareval = co.inputs[0].filename
            
            if '0133' in compareval:
                xshift = 0
                yshift = 0
            elif '0134' in compareval:
                xshift = 34.37
                yshift = -34.23
            elif '0135' in compareval:
                xshift = -34.31
                yshift = -33.78
            elif '0136' in compareval:
                xshift = 34.42
                yshift = 34.66
            elif '0137' in compareval:
                xshift = -34.35
                yshift = 34.33
            else:
                xshift = 0
                yshift = 0
            
            os.system( 'rm test.fits &> /dev/null' )
            outfile = os.path.basename( co.prependNames( 'shift_' )[0][0] )
            infile = co.inputsAsStr() 
            '''
            print 'INPUT:'
            print infile
            print 'XSHIFT:', xshift, 'YSHIFT:', yshift
            print 'OUTPUT:'
            print outfile
            '''
            images.imshift( infile + '[1]', output='test.fits', xshift=xshift, yshift=yshift)
            
            # This pyfits code is for dealing with the fact that imshift does not copy over the PHU of
            # the fits file.
            temp1 = pyfits.open( infile, 'readonly' )
            temp2 = pyfits.open( 'test.fits' )
            temp1[1].data = temp2[0].data
            os.system( 'rm ' + outfile + '  &> /dev/null' )
            temp1.writeto( outfile )
            co.reportOutput( co.prependNames("shift_") )
        except:
            print 'problem shifting image'
            raise
        yield co

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def findshiftsAndCombine(self, co):
       try:
          print "shifting and combining images"
          #@@TODO: hardcoded parmeters and ***imcoadd.dat may need to move from 
          # imcoadd_data/test4 to test_data dir before running
          gemini.imcoadd(co.stack_inputsAsStr(),fwhm=5, threshold=100,\
                fl_over=yes, fl_avg=yes,
                Stdout = co.getIrafStdout(), Stderr = co.getIrafStderr())
       except:
          print "Problem shifting and combining images "
          print "Problem in IMCOADD"
          raise
       yield co
       
    
#------------------------------------------------------------------------------ 
    def measureIQ(self, co):
        try:
            #@@FIXME: Detecting sources is done here as well. This should eventually be split up into
            # separate primitives, i.e. detectSources and measureIQ.
            print "measuring iq"
            import iqtool
            from iqtool.iq import getiq
            '''
            image, outFile='default', function='both', verbose=True,\
            residuals=False, display=True, \
            interactive=False, rawpath='.', prefix='auto', \
            observatory='gemini-north', clip=True, \
            sigma=2.3, pymark=True, niters=4, boxSize=2., debug=False):
            '''
            for inp in co.inputs:
                if 'GEMINI_NORTH' in inp.ad.getTypes():
                    observ = 'gemini-north'
                elif 'GEMINI_SOUTH' in inp.ad.getTypes():
                    observ = 'gemini-south'
                else:
                    observ = 'gemini-north'
                st = time.time()
                iqdata = getiq.gemiq( inp.filename, function='moffat', display=False, mosaic=True)
                et = time.time()
                print 'MeasureIQ time:', (et - st)
                # iqdata is list of tuples with image quality metrics
                # (ellMean, ellSig, fwhmMean, fwhmSig)
                co.rqIQ( inp.ad, *iqdata[0] )
            
        except:
            print 'Problem measuring IQ'
            raise 
        
        yield co
            
