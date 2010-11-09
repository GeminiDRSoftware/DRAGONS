import sys, StringIO, os
#from Reductionobjects import Reductionobject

import time
from astrodata.adutils import filesystem
from astrodata.adutils import gemLog
from astrodata import IDFactory
from astrodata import Descriptors
from astrodata.data import AstroData
from primitives_GEMINI import GEMINIPrimitives
from primitives_GEMINI import pyrafLoader
from primitives_GMOS import GMOSPrimitives
from gempy.instruments import geminiTools as gemt
from gempy.instruments import gmosTools as gmost
from devel.gmos import girmfringe
import pyfits as pf
#import numdisplay

log=gemLog.getGeminiLog()

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

class GMOS_IMAGEPrimitives(GMOSPrimitives):
    astrotype = "GMOS_IMAGE"
    
    def init(self, rc):
        
        GMOSPrimitives.init(self, rc)
        return rc

    def daverageCombine(self, rc):
        try:
            pyraf, gemini, yes, no = pyrafLoader(rc)
            
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
    
    def dbiasCorrect(self, rc):
        # not really sure we need to use gireduce here. I think we could easily make a
        # more generic bias sub task
        pyraf,gemini, yes, no = pyrafLoader(rc)
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
    
    def display(self, rc):
        from astrodata.adutils.future import gemDisplay
        pyraf, gemini, yes, no = pyrafLoader(rc)
        pyraf.iraf.set(stdimage='imtgmos')
        ds = gemDisplay.getDisplayService()
        for i in range(0, len(rc.inputs)):   
            inputRecord = rc.inputs[i]
            gemini.gmos.gdisplay( inputRecord.filename, i+1, fl_imexam=pyraf.iraf.no,
                Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr() )
            # this version had the display id conversion code which we'll need to redo
            # code above just uses the loop index as frame number
            #gemini.gmos.gdisplay( inputRecord.filename, ds.displayID2frame(rq.disID), fl_imexam=iraf.no,
            #    Stdout = coi.getIrafStdout(), Stderr = coi.getIrafStderr() )
        yield rc
    
    def findshiftsAndCombine(self, rc):
       try:
          pyraf,gemini, yes, no = pyrafLoader(rc)
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
    
    def dfringeCorrect(self, rc):
        try:
            pyraf, gemini, yes, no = pyrafLoader(rc)
            print "subtracting fringe frame"
            gemini.girmfringe( rc.inputsAsStr(), rc["fringe"],
                              Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr() )
            
            if gemini.girmfringe.status:
                raise GMOS_IMAGEException('girmfringe failed')
            
        except:
            print "Problem subtracting fringe from "+rc.inputsAsStr()
            raise 

        yield rc
        
    def fringeCorrect(self, rc):
        """
        This primitive will scale and subtract the fringe frame from the inputs.
        """
        
        try: 
            log.status('*STARTING* to fringe correct the images')
            
            # Using the same fringe file for all the input images
            #fringe = $$$$$$$$$$$$$$$$$$$$$$$$$$$ ??? where to get it  ?????
            #$$$$$$$ TEMP $$$$$$$$$$$$
            fringe = rc.getInputs(style='AD')[0]
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            
            for ad in rc.getInputs(style='AD'):
                
                # Loading up a dictionary with the input parameters for girmfringe
                paramDict = {
                             'inimage'        :ad,
                             'fringe'         :fringe,
                             'fl_statscale'   :rc['fl_statscale'],
                             'statsec'        :rc['statsec'],
                             'scale'          :rc['scale'],
                             'outpref'        :rc['postpend'],
                             }
                
                # Logging values set in the parameters dictionary above
                log.fullinfo('\nParameters being used for girmfringe '+
                             'function:\n')
                gemt.LogDictParams(paramDict)
                
                # Calling the girmfringe function to perform the fringe 
                # corrections, this function will return the corrected image as
                # an AstroData instance
                adOut = girmfringe.girmfringe(**paramDict)
                
                # Adding GEM-TLM(automatic) and RMFRINGE time stamps to the PHU     
                adOut.historyMark()
                adOut.historyMark(key='RMFRINGE', stomp=False)    
                
                log.fullinfo('************************************************'
                             ,'header')
                log.fullinfo('file = '+ad.filename, category='header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                             ,'header')
                log.fullinfo('PHU keywords updated/added:\n', category='header')
                log.fullinfo('GEM-TLM = '+adOut.phuGetKeyValue('GEM-TLM'), \
                             category='header')
                log.fullinfo('RMFRINGE = '+adOut.phuGetKeyValue('RMFRINGE'), category='header')
                log.fullinfo('------------------------------------------------'
                             , category='header')
                
                # Updating the file name with the postpend/outsuffix for this
                # primitive and then reporting the new file to the reduction 
                # context
                log.debug('Calling gemt.fileNameUpdater on '+ad.filename)
                adOut.filename = gemt.fileNameUpdater(ad.filename, 
                                                   postpend=rc['postpend'], 
                                                   strip=False)
                log.status('File name updated to '+adOut.filename)
                rc.reportOutput(adOut)        
                
            log.status('*FINISHED* adding the VAR frame(s) to the input data')
            
        except:
            print "Problem subtracting fringe from "+rc.inputsAsStr()
            raise 

        yield rc 
            
    def getForFringe(self, rc):
        """
        This primitive will check the files in the fringe lists are on disk,
        if not write them to disk and then update the inputs list to include
        all members of the fringe list for stacking.
        
        """
        try:
            log.status('*STARTING* to prepare the fringe list for stacking')
            # @@REFERENCE IMAGE @@NOTE: to pick which stackable list to get
            stackid = IDFactory.generateStackableID(rc.inputs[0].ad, 
                                                    stackType='FringeList')
            log.fullinfo('getting stack '+stackid, category='stack')
            # Getting the reduction context to update the appropriate cache 
            # files with the current inputs
            rc.rqStackGet(stackType='FringeList')
            # Yielding the reduction context so it can perform the 
            # requested updates
            yield rc
            # Retrieving the updated stack list
            stack = rc.getStack(stackid).filelist
            # Loading the file in the stack into memory
            print 'prim_G_I237: stack',stack
            rc.reportOutput(stack)
            
            log.status('*FINISHED* preparing the fringe list for stacking')
        except:
            log.critical('Problem getting stack '+stackid, category='stack')
            raise 
        yield rc
    
    def dmakeFringeFrame(self, rc):
        try:
            pyraf, gemini, yes, no = pyrafLoader(rc)
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
    
    def makeFringeFrame(self, rc):
        """
        This primitive will create a single fringe image from all the inputs.
        $$$$$$$$ MORE $$$$$$$$$$$$$$$$
        """
        # Loading and bringing the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader(rc)
        
        try:
            if len(rc.getInputs())>1:
                log.status('*STARTING* to create a fringe frame from the inputs')
    
                # Preparing input files, lists, parameters... for input to 
                # the CL script
                clm=gemt.CLManager(rc)
                clm.LogCurParams()
    
                # Creating a dictionary of the parameters set by the CLManager  
                # or the definition of the primitive 
                clPrimParams = {
                    # Retrieving the inputs as a list from the CLManager
                    'inimages'    :clm.inputList(),
                    # Maybe allow the user to override this in the future. 
                    'outimage'    :clm.combineOutname(), 
                    # This returns a unique/temp log file for IRAF  
                    'logfile'     :clm.logfile(),  
                    # This is actually in the default dict but wanted to 
                    # show it again       
                    'Stdout'      :gemt.IrafStdout(), 
                    # This is actually in the default dict but wanted to 
                    # show it again    
                    'Stderr'      :gemt.IrafStdout(),
                    # This is actually in the default dict but wanted to 
                    # show it again     
                    'verbose'     :yes                    
                              }
    
                # Creating a dictionary of the parameters from the Parameter 
                # file adjustable by the user
                clSoftcodedParams = {
                    'fl_vardq'      :gemt.pyrafBoolean(rc['fl_vardq']),
                    'combine'       :rc['method'],
                    'reject'        :'none',
                    'outpref'       :rc['postpend'],
                                    }
                # Grabbing the default parameters dictionary and updating 
                # it with the two above dictionaries
                clParamsDict = CLDefaultParamsDict('gifringe')
                clParamsDict.update(clPrimParams)
                clParamsDict.update(clSoftcodedParams)
                
                # Logging the values in the soft and prim parameter dictionaries
                log.fullinfo('\nParameters set by the CLManager or dictated by the'+
                             ' definition of the pritive:\n', category='parameters')
                gemt.LogDictParams(clPrimParams)
                log.fullinfo('\nUser adjustable parameters in the parameters '+
                             'file:\n', category='parameters')
                gemt.LogDictParams(clSoftcodedParams)
                
                log.debug('Calling the gifringe CL script for input list '+
                              clm.inputList())
                
                gemini.gifringe(**clParamsDict)
                
                if gemini.gifringe.status:
                    log.critical('gifringe failed for inputs '+rc.inputsAsStr())
                    raise GMOS_IMAGEException('gifringe failed')
                else:
                    log.status('Exited the gifringe CL script successfully')
                    
                # Renaming CL outputs and loading them back into memory 
                # and cleaning up the intermediate temp files written to disk
                clm.finishCL(combine=True) 
                #clm.rmStackFiles() #$$$$$$$$$ DON'T do this if 
                #^ Intermediate outputs are wanted!!!!
                
                # There is only one at this point so no need to perform a loop
                ad = rc.getOutputs(style='AD')[0] 
                
                # Adding a GEM-TLM (automatic) and FRINGE time stamps 
                # to the PHU
                ad.historyMark(key='FRINGE',stomp=False)
                # Updating logger with updated/added time stamps
                log.fullinfo('************************************************'
                             ,'header')
                log.fullinfo('file = '+ad.filename, 'header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                             , 'header')
                log.fullinfo('PHU keywords updated/added:\n', 'header')
                log.fullinfo('GEM-TLM = '+ad.phuGetKeyValue('GEM-TLM'), 
                             'header')
                log.fullinfo('COMBINE = '+ad.phuGetKeyValue('FRINGE'), 
                             'header')
                log.fullinfo('------------------------------------------------'
                             , 'header')        
                
                log.status('*FINISHED* creating the fringe image')
        except:
            print "Problem creating fringe from "+rc.inputsAsStr()
            raise 
        yield rc  
    
    def dmakeProcessedBias(self, rc):
        # Things done to the bias image before we subtract it:
        # overscan subtract
        # overscan trim
        # average combine images
    
        # BIAS made for all GMOS modes (imaging, spectroscopy, IFU) we need to
        # consider a generic task. using gbias (IRAF generic task)
        try:
            pyraf,gemini, yes, no = pyrafLoader(rc)
            print "combining biases to create master bias"
            gemini.gbias(rc.inputsAsStr(), outbias=rc["outbias"],
                rawpath=rc["caldir"], fl_trim=rc["fl_trim"], 
                fl_over=rc["fl_over"], fl_vardq=rc["fl_vardq"],
                Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr())
        except:
            print "Problem combining biases with gbias"
            raise
    
        yield rc 
    
    def dmakeProcessedFlat(self, rc):
        # FLAT made for all GMOS modes (imaging, spectroscopy, IFU) we need to
       
        try:
            pyraf,gemini, yes, no = pyrafLoader(rc)
            print "combining images and bias to create master flat"
            gemini.giflat(rc.inputsAsStr(), outflat=rc["outflat"],
                rawpath=rc["caldir"], fl_trim=rc["fl_trim"], 
                fl_over=rc["fl_over"], fl_vardq=rc["fl_vardq"],
                Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr())
        
        except:
            
            print "Problem combining flats with giflat"
            raise
    
        yield rc 

    def doverscanCorrect(self, rc):
        print "Performing Overscan Correct (overSub, overTrim)"
        try:
            pyraf,gemini, yes, no = pyrafLoader(rc)
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
    
    def doverscanSub(self, rc):
        try:
            pyraf,gemini, yes, no = pyrafLoader(rc)
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
    
    def doverscanTrim(self, rc):
        try:
            pyraf,gemini, yes, no = pyrafLoader(rc)
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
    
    def dprepare(self, rc):
        try:
            pyraf,gemini, yes, no = pyrafLoader(rc)
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
        
    def setForFringe(self, rc):
        """ 
        This primitive will update the list of files with the same observationID
        to be combined into a single fringe frame. This list file is cached 
        between calls to reduce, thus allowing for one-file-at-a-time processing. 
        The fringe file will be produced with the makeFringe primitive and
        later used to perform the fringe correction of the input science images 
        in the fringeCorrect primitive. 
        """
        try:
            log.status('*STARTING* to update/create the fringe list')
            for ad in rc.getInputs(style='AD'):
                fringeID = IDFactory.generateAstroDataID(ad)
                print 'fringeID: ',repr(fringeID)
                listID = IDFactory.generateFringeListID(ad)
                print 'listID: ',repr(listID)
                
                rc.fringes.add(listID, fringeID, ad)
                rc.rqStackUpdate(stackType='FringeList')
                print 'prim_G_I390: ',rc.inputsAsStr()
                # Writing the files in the stack to disk if not all ready there
                if not os.path.exists(ad.filename):
                    log.fullinfo('temporarily writing '+ad.filename+\
                                 ' to disk', category='stack')
                    ad.write(ad.filename)
                    
            log.status('*FINISHED* updating/creating the fringe list')
        except:
            raise 
        
        yield rc
    
    def shift(self, rc):
        '''
        !!!NOTE!!!
        The code in this method was designed only for demo use. 
        It should not be taken seriously.
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
                
                # This pf code is for dealing with the fact that imshift does not copy over the PHU of
                # the fits file.
                temp1 = pf.open( infile, 'readonly' )
                temp2 = pf.open( tmpname )
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

def CLDefaultParamsDict(CLscript):
    """
    A function to return a dictionary full of all the default parameters 
    for each CL script used so far in the Recipe System.
    
    """
    # loading and bringing the pyraf related modules into the name-space
    pyraf, gemini, yes, no = pyrafLoader()
    
    # Ensuring that if a invalide CLscript was requested, that a critical
    # log message be made and exception raised.
    if CLscript != 'gifringe':
        log.critical('The CLscript '+CLscript+' does not have a default'+
                     ' dictionary')
        raise GEMINIException('The CLscript '+CLscript+
                              ' does not have a default'+' dictionary')
        
    if CLscript == 'gifringe':
        defaultParams = {
            'inimages'  :'',              # Input GMOS images
            'outimage'  : '',             # Output fringe frame
            'typezero'  : 'mean',         # Operation to determine the sky level or zero point
            'skysec'    : 'default',      # Zero point statistics section
            'skyfile'   : '',             # File with zero point values for each input image
            'key_zero'  : 'OFFINT',       # Keyword for zero level
            'msigma'    : 4.0,            # Sigma threshold above sky for mask
            'bpm'       : '',             # Name of bad pixel mask file or image
            'combine'   : 'median',       # Combination operation
            'reject'    : 'avsigclip',    # Rejection algorithm
            'scale'     : 'none',         # Image scaling
            'weight'    : 'none',         # Image Weights
            'statsec'   : '[*,*]',        # Statistics section for image scaling
            'expname'   : 'EXPTIME',      # Exposure time header keyword for image scaling
            'nlow'      : 1,              # minmax: Number of low pixels to reject
            'nhigh'     : 1,              # minmax: Number of high pixels to reject
            'nkeep'     : 0,              # Minimum to keep or maximum to reject
            'mclip'     : yes,            # Use median in sigma clipping algorithms?
            'lsigma'    : 3.0,            # Lower sigma clipping factor
            'hsigma'    : 3.0,            # Upper sigma clipping factor
            'sigscale'  : 0.1,            # Tolerance for sigma clipping scaling correction
            'sci_ext'   : 'SCI',          # Name of science extension
            'var_ext'   : 'VAR',          # Name of variance extension
            'dq_ext'    : 'DQ',           # Name of data quality extension
            'fl_vardq'  : no,             # Make variance and data quality planes?
            'logfile'   : '',             # Name of the logfile
            'glogpars'  : '',             # Logging preferences
            'verbose'   : yes,            # Verbose output
            'status'    : 0,              # Exit status (0=good)
            'Stdout'    :gemt.IrafStdout(),
            'Stderr'    :gemt.IrafStdout()
                       }
        return defaultParams
