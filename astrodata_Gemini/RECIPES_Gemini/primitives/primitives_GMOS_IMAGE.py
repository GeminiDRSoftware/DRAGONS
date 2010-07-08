#from Reductionobjects import Reductionobject
from primitives_GEMINI import GEMINIPrimitives
# All GEMINI IRAF task wrappers.
import time
from astrodata.adutils import filesystem
from astrodata.adutils.future import gemLog
from astrodata import IDFactory
from astrodata import Descriptors
from astrodata.data import AstroData

from pyraf.iraf import tables, stsdas, images
from pyraf.iraf import gemini
import pyraf
import iqtool
from iqtool.iq import getiq
from gempy.instruments.gemini import *
from gempy.instruments.gmos import *

import pyfits
import numdisplay
import string
log=gemLog.getGeminiLog()
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
    astrotype = "GMOS_IMAGE"
    
    def init(self, rc):
        
        if "iraf" in rc and "adata" in rc["iraf"]:
            pyraf.iraf.set (adata=rc["iraf"]['adata'])  
        else:
            # @@REFERENCEIMAGE: used to set adata path for primitives
            if len(rc.inputs) > 0:
                (root, name) = os.path.split(rc.inputs[0].filename)
                pyraf.iraf.set (adata=root)
                if "iraf" not in rc:
                    rc.update({"iraf":{}})
                if "adata" not in rc["iraf"]:
                    rc["iraf"].update({"adata":root}) 
        
        GEMINIPrimitives.init(self, rc)
        return rc
       
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def averageCombine(self, rc):
        try:
            # @@TODO: need to include parameter options here
            print "Combining and averaging" 
            filesystem.deleteFile('inlist')
            
            tempset = set()
            
            for inp in rc.inputs:
                 tempset.add( IDFactory.generateStackableID( inp.ad ) )
                 
            for stackID in tempset:
                #@@FIXME: There are a lot of big issues in here. First, we need to backup the
                # previous average combined file, not delete. Backup is needed if something goes wrong.
                # Second, the pathnames in here have too many assumptions. (i.e.) It is assumed all the
                # stackable images are in the same spot which may not be the case.
                
                stacklist = [ os.path.basename(f) 
                                for f in rc.getStack(stackID).filelist ]
                
                #print "pG147: STACKLIST:", stacklist

                
                if len( stacklist ) > 1:
                    # @@REFERENCEIMAGE: first image used as reference image to
                    # creat the output filename
                    stackname = "avgcomb_" + os.path.basename(stacklist[0])
                    inlistname = "inlist."+stackID
                    
                    # slight kludge, remove output if it already exists from another run
                    # since it always uses the zeroeth image
                    if (os.path.exists(stackname)):
                        os.remove(stackname)
                        
                    gemini.gemcombine( rc.makeInlistFile(inlistname, stacklist),  output=stackname,
                       combine="average", reject="none" ,Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr())
                    
                    if gemini.gemcombine.status:
                        raise GMOS_IMAGEException('gemcombine failed')
                    
                    rc.reportOutput(stackname)
                else:
                    print "'%s' was not combined because there is only one image." %( stacklist[0] )
        except:
            raise
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
                # print "pGI:", ",".join(cals[cal])
                # gireduce was taking the join above as it's inputs... not sure why
                # so leaving this note
                gemini.gmos.gireduce(rc.inputsAsStr(), fl_over=no,
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
    
    def display(self, rc):
        from astrodata.adutils.future import gemDisplay
        import pyraf
        from pyraf import iraf
        iraf.set(stdimage='imtgmos')
        ds = gemDisplay.getDisplayService()
        for i in range(0, len(rc.inputs)):   
            inputRecord = rc.inputs[i]
            gemini.gmos.gdisplay( inputRecord.filename, i+1, fl_imexam=iraf.no,
                Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr() )
            # this version had the display id conversion code which we'll need to redo
            # code above just uses the loop index as frame number
            #gemini.gmos.gdisplay( inputRecord.filename, ds.displayID2frame(rq.disID), fl_imexam=iraf.no,
            #    Stdout = coi.getIrafStdout(), Stderr = coi.getIrafStderr() )
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
                # see Bias correct, used to have strange join from cal structures
                # instead of rc.inputsAsStr() below.
                gemini.gmos.gireduce(rc.inputsAsStr(), fl_over=no,fl_trim=no,
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
                
                xoffset = inp.ad.x_offset()
                pixscale = inp.ad.pixel_scale()
                yoffset = inp.ad.y_offset()
                
                if xoffset and pixscale and yoffset:
                    xshift = xoffset / pixscale * -1
                    yshift = yoffset / pixscale * -1
                else:
                    print "${RED}ERROR: insufficient information to shift (set PIXSCALE, XOFFSET, and YOFFSET in PHU)"
                    xshift = 0
                    yshift = 0                   
                
                #os.system( 'rm test.fits &> /dev/null' )
                infile = inp.filename
                outfile =  "shift_"+ os.path.basename(infile)
                
                #print 'INPUT:'
                #print infile
                print 'XSHIFT:', xshift, 'YSHIFT:', yshift
                #print 'OUTPUT:'
                #print outfile
                
                tmpname = "tmpshift_"+ os.path.basename(infile) + ".fits"
                images.imshift( infile + '[1]', output= tmpname, xshift=xshift, yshift=yshift)
                
                # This pyfits code is for dealing with the fact that imshift does not copy over the PHU of
                # the fits file.
                temp1 = pyfits.open( infile, 'readonly' )
                temp2 = pyfits.open( tmpname )
                temp1[1].data = temp2[0].data
                #os.system( 'rm ' + outfile + '  &> /dev/null' )
                temp1.writeto( outfile )
                temp1.close()
                temp2.close()
                os.remove(tmpname)
            rc.reportOutput( rc.prependNames("shift_") )
        except:
            print 'Problem shifting image'
            raise 

        yield rc
     #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   
    def prepare(self, rc):
        try:
            print 'preparing'
            print "Updating keywords PIXSCALE, NEXTEND, OBSMODE, GEM-TLM, GPREPARE"
            print "Updating GAIN keyword by calling GGAIN"
            
            print "pGI391:",repr(rc)
            gemini.gmos.gprepare(rc.inputsAsStr(strippath = True), rawpath=rc['iraf']['adata'],
                                 Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr())
            #
            if gemini.gmos.gprepare.status:
                raise GMOS_IMAGEException( 'gprepare failed')
            
            rc.reportOutput(rc.prependNames("g", currentDir = True))
            
        except:
            print "Problem preparing the image."
            raise 
        
        yield rc    
        
    #$$$$$$$$$$$$$$$$$$$$ NEW STUFF BY KYLE FOR: PREPARE $$$$$$$$$$$$$$$$$$$$$
    '''
    these primitives are now functioning and can be used, BUT are not set up to run with the current demo system.
    commenting has been added to hopefully assist those reading the code.
    excluding validateWCS, all the primitives for prepare are complete (as far as we know of at the moment that is)
    and so I am moving onto working on the primitives following prepare.
    '''
    
    def standardizeInstrumentHeaders(self,rc):
        '''
        makes the changes and additions to the headers of the input files that are instrument specific
        '''
        
        try:      
            writeInt = rc['writeInt']
                               
            for ad in rc.getInputs(style="AD"): 
                log.debug('calling stdInstHdrs','status') 
                stdInstHdrs(ad) 
            
            if writeInt:
                log.debug('writing the outputs to disk')
                rc.run('writeOutputs(postpend=_instHdrs)')  #$$$$$$$$$$$$$this needs to accept arguments to work right!!!!!!!!!!!! currently hardcoded
                log.debug('writting complete')
                    
        except:
            log.critical("Problem preparing the image.",'critical')
            raise 
        
        yield rc
    #------------------------------------------------------------------------    
    def validateInstrumentData(self,rc):
        '''
        instrument specific validations for the input file
        '''
        
        try:
            for ad in rc.getInputs(style="AD"):
                log.status('validating data for file = '+ad.filename,'status')
                log.debug('calling valInstData', 'status')
                valInstData(ad)
                #log.status('data validated for file = '+ad.filename,'status')
                
        except:
            log.critical("Problem preparing the image.",'critical')
            raise 
        
        yield rc       
 #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Prepare primitives end here $$$$$$$$$$$$$$$$$$$$$$$$$$$$

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ primitives following Prepare below $$$$$$$$$$$$$$$$$$$$  
    def overscanSubtract(self,rc):
        try:
            log.status('STARTING to subtract the overscan from the input data')
            ## writing input files to disk with prefixes onto their file names so they can be deleted later easily 
            clm = CLManager(rc)
            curprefix=clm.uniquePrefix()

            ## params in the dictionaries: fl_over, fl_trim, fl_vardq, outpref
            log.fullinfo('calling the gireduce CL script', 'status')
            gemini.gmos.gireduce(clm.inputsAsStr(), gp_outpref=curprefix,fl_over=pyrafBoolean(rc["fl_over"]), \
                    fl_trim=pyrafBoolean(rc["fl_trim"]), fl_bias=no, \
                    fl_flat=no, outpref=rc["outpref"], fl_vardq=pyrafBoolean(rc['fl_vardq']),\
                    Stdout = "dev$null", Stderr = rc.getIrafStderr())
            #"dev$null" #use this for Stdout for no outputs of the CL script to go to screen
            #rc.getIrafStdout() #us this for Stdout for outputs of the CL script to go to screen
            if gemini.gmos.gireduce.status:
                log.critical('gireduce failed','critical') 
                raise
            else:
                log.fullinfo('exited the gireduce CL script successfully', 'status')
         
            # renaming CL outputs and loading them back into memory and cleaning up the intermediate tmp files written to disk
            clm.finishCL()
            i=0
            for ad in rc.getOutputs(style="AD"):
                if ad.phuGetKeyValue('GIREDUCE'): # varifies gireduce was actually ran on the file
                    log.fullinfo('file '+clm.preCLNames()[i]+' had its overscan subracted successfully', 'status')
                    log.fullinfo('new file name is: '+ad.filename, 'status')
                i=i+1
                ut = datetime.now().isoformat()  
                ad.phuSetKeyValue('GEM-TLM', ut , 'UT Last modification with GEMINI')
                
                log.fullinfo('****************************************************','header')
                log.fullinfo('file = '+ad.filename,'header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+str(ut)+'/n','header' )
            
            log.status('FINISHED subtracting the overscan from the input data','status')
        except:
            log.critical("Problem preparing the image.",'critical')
            raise 
        
        yield rc    
        
    def overscanTrim(self,rc):
        try:
            log.status('STARTING to trim the overscan region from the input data','status')
            
            for ad in rc.getInputs(style='AD'):
                for sciExt in ad['SCI']:
                    datasecStr=sciExt.data_section()
                    datasecList=secStrToIntList(datasecStr) 
                    dsl=datasecList
                    log.stdinfo('\nfor '+ad.filename+' extension '+str(sciExt.extver())+\
                                                            ', keeping the data from the section '+datasecStr,'science')
                    sciExt.data=sciExt.data[dsl[2]-1:dsl[3],dsl[0]-1:dsl[1]]
                    sciExt.header['NAXIS1']=dsl[1]-dsl[0]+1
                    sciExt.header['NAXIS2']=dsl[3]-dsl[2]+1

                    ## updating logger with updated/added keywords to each SCI frame
                    log.fullinfo('****************************************************','header')
                    log.fullinfo('file = '+ad.filename,'header')
                    log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','header')
                    log.fullinfo('SCI extension number '+str(sciExt.extver())+' keywords updated/added:\n', 'header')
                    log.fullinfo('NAXIS1= '+str(sciExt.header['NAXIS1']),'header' )
                    log.fullinfo('NAXIS2= '+str(sciExt.header['NAXIS2']),'header' )
                    
                # updating the GEM-TLM value and reporting the output to the RC    
                ut = datetime.now().isoformat()  
                ad.phuSetKeyValue('GEM-TLM', ut , 'UT Last modification with GEMINI')
                ad.filename=fileNameUpdater(ad.filename,postpend='_overtrimd', strip=False)
                rc.reportOutput(ad)
                
                # updating logger with updated/added keywords to the PHU
                log.fullinfo('****************************************************','header')
                log.fullinfo('file = '+ad.filename,'header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+str(ut)+'/n','header' ) 
            log.status('FINISHED trimming the overscan region from the input data','status')
        except:
            log.critical("Problem preparing the image.",'critical')
            raise 
        
        yield rc

def pyrafBoolean(pythonBool):
    '''
    a very basic function to reduce code repetition that simply 'casts' any given 
    Python boolean into a pyraf/iraf one for use in the CL scripts
    '''
    
    if pythonBool:
        return pyraf.iraf.yes
    elif  not pythonBool:
        return pyraf.iraf.no
    else:
        print "DANGER DANGER Will Robinson, pythonBool passed in not True or False"

    #$$$$$$$$$$$$$$$$$$$$$$$ END OF KYLES NEW STUFF $$$$$$$$$$$$$$$$$$$$$$$$$$
        
