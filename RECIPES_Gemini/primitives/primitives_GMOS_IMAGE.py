#from Reductionobjects import Reductionobject
from primitives_GEMINI import GEMINIPrimitives
# All GEMINI IRAF task wrappers.
import time
from utils import filesystem
from astrodata import IDFactory
from astrodata import Descriptors
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

class GMOS_IMAGEException:
    """ This is the general exception the classes and functions in the
    Structures.py module raise.
    """
    def __init__(self, msg="Exception Raised in Recipe System"):
        """This constructor takes a message to print to the user."""
        self.message = msg
    def __str__(self):
        """This str conversion member returns the message given by the user (or the default message)
        when the exception is not caught."""
        return self.message


class GMOS_IMAGEPrimitives(GEMINIPrimitives):
    def init(self, rc):
        if "global" in rc and "adata" in rc["global"]:
            pyraf.iraf.set (adata=rc["global"]['adata'].value)  
        else:
            (root, name) = os.path.split(rc.inputs[0])
            pyraf.iraf.set (adata=root)  
        
        GEMINIPrimitives.init(self, rc)
        return rc
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   
    def averageCombine(self, rc):
        try:
            # @@TODO: need to include parameter options here
            print "Combining and averaging" 
            filesystem.deleteFile('inlist')
            
            templist = []
            
            for inp in rc.inputs:
                 templist.append( IDFactory.generateStackableID( inp.ad ) )
                 
            templist = list( set(templist) ) # Removes duplicates.
            
            for stackID in templist:
                #@@FIXME: There are a lot of big issues in here. First, we need to backup the
                # previous average combined file, not delete. Backup is needed if something goes wrong.
                # Second, the pathnames in here have too many assumptions. (i.e.) It is assumed all the
                # stackable images are in the same spot which may not be the case.
                
                stacklist = rc.getStack( stackID ).filelist
                #print "pG147: STACKLIST:", stacklist

                
                if len( stacklist ) > 1:
                    stackname = "avgcomb_" + os.path.basename(stacklist[0])
                    filesystem.deleteFile( stackname )
                    gemini.gemcombine( rc.makeInlistFile(stackID),  output=stackname,
                       combine="average", reject="none" ,Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr())
                    
                    if gemini.gemcombine.status:
                        raise GMOS_IMAGEException('gemcombine failed')
                    
                    rc.reportOutput(stackname)
                else:
                    print "'%s' was not combined because there is only one image." %( stacklist[0] )
        except:
            raise GMOS_IMAGEException("Problem combining and averaging")

        yield rc
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def biasCorrect(self, rc):
        # not really sure we need to use gireduce here. I think we could easily make a
        # more generic bias sub task
        try:
            print "Subtracting off bias"
            cals = rc.calFilename( 'bias' )
            for cal in cals:
                gemini.gmos.gireduce(",".join(cals[cal]), fl_over=no,
                    fl_trim=no, fl_bias=yes,bias=cal,
                    fl_flat=no, outpref="biassub_",
                    Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr()
                    ) # this flag was removed?,fl_mult=no)
            
            if gemini.gmos.gireduce.status:
                raise GMOS_IMAGEException('gireduce failed')
            
            rc.reportOutput(rc.prependNames("biassub_"))
        except:
            print "Problem subtracting bias"
            raise
            
        yield rc
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def findshiftsAndCombine(self, rc):
       try:
          print "shifting and combining images"
          #@@TODO: hardcoded parmeters and ***imcoadd.dat may need to move from 
          # imcoadd_data/test4 to test_data dir before running
          gemini.imcoadd(rc.stack_inputsAsStr(),fwhm=5, threshold=100,\
                fl_over=yes, fl_avg=yes,
                Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr())
       except:
           print "Problem shifting and combining images"
           raise

       yield rc
       
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   
    def flatCreate(self, rc):

        # FLAT made with giflat
        try:
            print 'combining and normalizing best 20 twilight flats'
            gemini.giflat(rc.inputsAsStr(), outflat=rc["outflat"],
                bias=rc.calName("REDUCED_BIAS"),rawpath=rc["caldir"],
                fl_over=rc["fl_over"], fl_trim=rc["fl_trim"], 
                fl_vardq=rc["fl_vardq"],Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr())
        except:
            print "Problem combining imaging flats with giflat"
            raise 
        
        yield rc
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def flatfieldCorrect(self, rc):
        try:
            print "Flat field correcting"
            
            cals = rc.calFilename( 'twilight' )
            for cal in cals:
                gemini.gmos.gireduce(",".join(cals[cal]), fl_over=no,fl_trim=no,
                    fl_bias=no, flat1=cal, fl_flat=yes, outpref="flatdiv_",
                    Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr())
            
            if gemini.gmos.gireduce.status:
                raise GMOS_IMAGEException('gireduce failed')
            
            rc.reportOutput(rc.prependNames("flatdiv_"))   
        except:
            print 'Problem dividing by normalized flat'
            raise 

        yield rc
        
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def fringeCorrect(self, rc):
        try:
            print "subtracting fringe frame"
            gemini.girmfringe( rc.inputsAsStr(), rc["fringe"],
                              Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr() )
            
            if gemini.girmfringe.status:
                raise GMOS_IMAGEException('girmfringe failed')
            
        except:
            print "Problem subtracting fringe from "+rc.inputsAsStr()
            raise 

        yield rc
        
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def makeFringeFrame(self, rc):
        try:
            print "creating fringe frame"
            gemini.gifringe(rc.inputsAsStr(), "fringe",
                Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr())
            
            if gemini.gifringe.status:
                raise GMOS_IMAGEException('gifringe failed')
            
            # Not sure where the output is put...If the output is reported
            # then the inputs are no longer valid...and the fringe becomes the
            # inputs.
        except:
            print "Problem creating fringe from "+rc.inputsAsStr()
            raise 

        yield rc  
         
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def makeProcessedBias(self, rc):
        # Things done to the bias image before we subtract it:
        # overscan subtract
        # overscan trim
        # average combine images
    
        # BIAS made for all GMOS modes (imaging, spectroscopy, IFU) we need to
        # consider a generic task. using gbias (IRAF generic task)
        try:
            print "combining biases to create master bias"
            gemini.gbias(rc.inputsAsStr(), outbias=rc["outbias"],
                rawpath=rc["caldir"], fl_trim=rc["fl_trim"], 
                fl_over=rc["fl_over"], fl_vardq=rc["fl_vardq"],
                Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr())
        except:
            print "Problem combining biases with gbias"
            raise
    
        yield rc 

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def makeProcessedFlat(self, rc):
        # FLAT made for all GMOS modes (imaging, spectroscopy, IFU) we need to
       
        try:
            print "combining images and bias to create master flat"
            gemini.giflat(rc.inputsAsStr(), outflat=rc["outflat"],
                rawpath=rc["caldir"], fl_trim=rc["fl_trim"], 
                fl_over=rc["fl_over"], fl_vardq=rc["fl_vardq"],
                Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr())
        
        except:
            
            print "Problem combining flats with giflat"
            raise
    
        yield rc 
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def mosaicChips(self, rc):
       try:
          print "producing image mosaic"
          gemini.gmosaic( rc.inputsAsStr(), outpref="mo_",
            Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr() )
          
          if gemini.gmosaic.status:
              raise GMOS_IMAGEException('gmosaic failed')
          
          rc.reportOutput(rc.prependNames("mo_", currentDir = True))
       except:
           print "Problem producing image mosaic"
           raise 

       yield rc
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
    def overscanCorrect(self, rc):
        print "Performing Overscan Correct (overSub, overTrim)"
        try:
            gemini.gmos.gireduce(rc.inputsAsStr(strippath=True), fl_over=pyraf.iraf.yes,fl_trim=pyraf.iraf.yes,
                fl_bias=no, fl_flat=no, outpref="trim_oversub_",
                Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr())
            
            if gemini.gmos.gireduce.status:
                raise GMOS_IMAGEException('gireduce failed')
            
            rc.reportOutput(rc.prependNames("trim_oversub_", currentDir = True))
        except:
            print "Problem correcting overscan region"
            raise
        
        yield rc
        
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def overscanSub(self, rc):
        try:
            print "Determining overscan subtraction region using nbiascontam"
            print "parameter and BIASSEC header keyword"
            print "Subtracting overscan bias levels using colbias"
            gemini.gmos.gireduce(rc.inputsAsStr(strippath=True), fl_over=pyraf.iraf.yes,fl_trim=no, fl_bias=no, \
                fl_flat=no, outpref="oversub_",Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr())
            
            if gemini.gmos.gireduce.status:
                raise GMOS_IMAGEException('gireduce failed')
            
            rc.reportOutput(rc.prependNames("oversub_", currentDir = True))
        except:
            print "Problem subtracting overscan bias"
            raise 
        
        yield rc
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def overscanTrim(self, rc):
        try:
            print "Determining overscan region using BIASSEC header keyword"
            print "Trimming off overscan"
            gemini.gmos.gireduce(rc.inputsAsStr(), fl_over=no,fl_trim=yes, 
                fl_bias=no, fl_flat=no, outpref="trim_",Stdout = rc.getIrafStdout(),
                Stderr = rc.getIrafStderr())
            
            if gemini.gmos.gireduce.status:
                raise GMOS_IMAGEException('gireduce failed')
            
            rc.reportOutput(rc.prependNames("trim_"))
        except:
            print "Problem trimming off overscan region"
            raise 
            
        yield rc
        
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   
    def prepare(self, rc):
        try:
            print 'preparing'
            print "Updating keywords PIXSCALE, NEXTEND, OBSMODE, GEM-TLM, GPREPARE"
            print "Updating GAIN keyword by calling GGAIN"
            
            gemini.gmos.gprepare(rc.inputsAsStr(strippath = True), rawpath=rc['global']['adata'].value,
                                 Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr())
            
            if gemini.gmos.gprepare.status:
                raise GMOS_IMAGEException( 'gprepare failed')
            
            rc.reportOutput(rc.prependNames("g", currentDir = True))
            
        except:
            print "Problem preparing the image."
            raise 
        
        yield rc 
        
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def setForFringe(self, rc):
        try:
            print 'adding to fringe list'
            for inp in rc.inputs:
                fringeID = IDFactory.generateAstroDataID( inp )
                listID = IDFactory.generateFringeListID( inp )
                rc.fringes.add( listID, fringeID, inp )
        except:
            raise 
        
        yield rc
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def shift(self, rc):
        '''
        !!!NOTE!!!
        The code in this method was designed only for demo use. It should not be taken seriously.
        '''
        try:
            print 'shifting image'
            for inp in rc.inputs:
                '''
                compareval = inp.filename
                
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
                '''
                
                #===================================================================
                # Simple code that uses instrument sensors values from header to determine
                # x and y shift offsets. No rotation stuff put in. Use at own risk!
                # -River
                #===================================================================
                
                xoffset = inp.ad.xoffset()
                pixscale = inp.ad.pixscale()
                yoffset = inp.ad.yoffset()
                
                if xoffset and pixscale and yoffset:
                    xshift = xoffset / pixscale * -1
                    yshift = yoffset / pixscale * -1
                else:
                    print "${RED}ERROR: insufficient information to shift (set PIXSCALE, XOFFSET, and YOFFSET in PHU)"
                    return
                
                os.system( 'rm test.fits &> /dev/null' )
                outfile = os.path.basename( rc.prependNames( 'shift_',  )[0][0] )
                infile = inp.filename
                
                #print 'INPUT:'
                #print infile
                print 'XSHIFT:', xshift, 'YSHIFT:', yshift
                #print 'OUTPUT:'
                #print outfile
                
                images.imshift( infile + '[1]', output='test.fits', xshift=xshift, yshift=yshift)
                
                # This pyfits code is for dealing with the fact that imshift does not copy over the PHU of
                # the fits file.
                temp1 = pyfits.open( infile, 'readonly' )
                temp2 = pyfits.open( 'test.fits' )
                temp1[1].data = temp2[0].data
                os.system( 'rm ' + outfile + '  &> /dev/null' )
                temp1.writeto( outfile )
                rc.reportOutput( rc.prependNames("shift_") )
        except:
            print 'Problem shifting image'
            raise 

        yield rc

   