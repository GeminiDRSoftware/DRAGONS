import sys, StringIO, os

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
from gempy.instruments import girmfringe

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
    """
        This is the class of all primitives for the GMOS_IMAGE level of the type 
        hierarchy tree.  It inherits all the primitives to the level above
        , 'GMOSPrimitives'.
    """
    astrotype = "GMOS_IMAGE"
    
    def init(self, rc):
        
        GMOSPrimitives.init(self, rc)
        return rc
        
    def fringeCorrect(self, rc):
        """
        This primitive will scale and subtract the fringe frame from the inputs.
        It utilizes the Python re-written version of cl script girmfringe to 
        do the work.
        
        :param postpend: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type postpend: string
        
        :param fl_statscale: Scale by statistics rather than exposure time
        :type fl_statscale: Boolean
        
        :param statsec: image section used to determine the scale factor 
                        if fl_statsec=True
        :type statsec: string of format '[EXTNAME,EXTVER][x1:x2,y1:y2]'
                       default: If CCDSUM = '1 1' :[SCI,2][100:1900,100:4500]'
                       If CCDSUM = '2 2' : [SCI,2][100:950,100:2250]'
        
        :param scale: Override auto-scaling if not 0.0
        :type scale: real
        
        :param logVerbose: Verbosity setting for log messages to the screen.
        :type logVerbose: int. 
                          This value can be set for each primitive individually 
                          in a recipe only (ie. not in the parameter file). 
                          If no value is specified during the recipe, the value 
                          set during the call to reduce or its default (2) will 
                          be used.
        """
        log = gemLog.getGeminiLog(verbose=int(rc['logVerbose']))
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
                             }
                
                # Logging values set in the parameters dictionary above
                log.fullinfo('\nParameters being used for girmfringe '+
                             'function:\n')
                gemt.logDictParams(paramDict)
                
                # Calling the girmfringe function to perform the fringe 
                # corrections, this function will return the corrected image as
                # an AstroData instance
                adOut = girmfringe.girmfringe(**paramDict)
                
                # Adding GEM-TLM(automatic) and RMFRINGE time stamps to the PHU     
                adOut.historyMark(key='RMFRINGE', stomp=False)    
                
                log.fullinfo('************************************************'
                             ,'header')
                log.fullinfo('file = '+ad.filename, category='header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                             ,'header')
                log.fullinfo('PHU keywords updated/added:\n', category='header')
                log.fullinfo('GEM-TLM = '+adOut.phuGetKeyValue('GEM-TLM'), 
                             category='header')
                log.fullinfo('RMFRINGE = '+adOut.phuGetKeyValue('RMFRINGE'), 
                             category='header')
                log.fullinfo('------------------------------------------------'
                             , category='header')
                
                # Updating the file name with the postpend/outsuffix for this
                # primitive and then reporting the new file to the reduction 
                # context
                log.debug('Calling gemt.fileNameUpdater on '+ad.filename)
                adOut.filename = gemt.fileNameUpdater(adIn=ad, 
                                                   postpend=rc['postpend'], 
                                                   strip=False)
                log.status('File name updated to '+adOut.filename)
                rc.reportOutput(adOut)        
                
            log.status('*FINISHED* fringe correcting the input data')
            
        except:
            log.critical("Problem subtracting fringe from "+rc.inputsAsStr())
            raise 

        yield rc 
    
    def makeFringeFrame(self, rc):
        """
        This primitive will create a single fringe image from all the inputs.
        It utilizes the CL script gifringe to create the fringe image.
        
        :param postpend: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type postpend: string
        
        :param fl_vardq: Create variance and data quality frames?
        :type fl_vardq: Python boolean (True/False), OR string 'AUTO' to do 
                        it automatically if there are VAR and DQ frames in the inputs.
                        NOTE: 'AUTO' uses the first input to determine if VAR and DQ frames exist, 
                        so, if the first does, then the rest MUST also have them as well.
        
        :param method: type of combining method to use on the fringe frames.
        :type method: string, options: 'average', 'median
        
        :param logVerbose: Verbosity setting for log messages to the screen.
        :type logVerbose: int. 
                          This value can be set for each primitive individually 
                          in a recipe only (ie. not in the parameter file). 
                          If no value is specified during the recipe, the value 
                          set during the call to reduce or its default (2) will 
                          be used.
        """
        log = gemLog.getGeminiLog(verbose=int(rc['logVerbose']))
        
        # Loading and bringing the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader()
        
        try:
            if len(rc.getInputs())>1:
                log.status('*STARTING* to create a fringe frame from the inputs')
    
                # Preparing input files, lists, parameters... for input to 
                # the CL script
                clm=gemt.CLManager(adIns=rc.getInputs(style='AD'), 
                                   postpend=rc['postpend'],  
                                   funcName='makeFringeFrame', 
                                   verbose=int(rc['logVerbose']))
    
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
                    'Stdout'      :gemt.IrafStdout(verbose=int(rc['logVerbose'])), 
                    # This is actually in the default dict but wanted to 
                    # show it again    
                    'Stderr'      :gemt.IrafStdout(verbose=int(rc['logVerbose'])),
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
                clParamsDict = CLDefaultParamsDict('gifringe', verbose=int(rc['logVerbose']))
                clParamsDict.update(clPrimParams)
                clParamsDict.update(clSoftcodedParams)
                
                # Logging the values in the soft and prim parameter dictionaries
                log.fullinfo('\nParameters set by the CLManager or dictated by '+
                         'the definition of the primitive:\n', 
                         category='parameters')
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
                ad = clm.finishCL(combine=True)  
                
                # Adding a GEM-TLM (automatic) and FRINGE time stamps 
                # to the PHU
                ad.historyMark(key='FRINGE',stomp=False)
                # Updating logger with updated/added time stamps
                log.fullinfo('************************************************'
                             ,'header')
                log.fullinfo('file = '+ad.filename, category='header')
                log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                             , 'header')
                log.fullinfo('PHU keywords updated/added:\n', category='header')
                log.fullinfo('GEM-TLM = '+ad.phuGetKeyValue('GEM-TLM'), 
                             category='header')
                log.fullinfo('FRINGE = '+ad.phuGetKeyValue('FRINGE'), 
                             category='header')
                log.fullinfo('------------------------------------------------'
                             , category='header')        
                
                log.status('*FINISHED* creating the fringe image')
        except:
            log.critical("Problem creating fringe from "+rc.inputsAsStr())
            raise 
        yield rc  

def CLDefaultParamsDict(CLscript, verbose=1):
    """
    A function to return a dictionary full of all the default parameters 
    for each CL script used so far in the Recipe System.
    
    :param verbose: Verbosity setting for log messages to the screen.
    :type logVerbose: int. Default: 1
    """
    log = gemLog.getGeminiLog(verbose=verbose)
    
    # loading and bringing the pyraf related modules into the name-space
    pyraf, gemini, yes, no = pyrafLoader()
    
    # Ensuring that if a invalide CLscript was requested, that a critical
    # log message be made and exception raised.
    if (CLscript != 'gifringe'):
        log.critical('The CLscript '+CLscript+' does not have a default'+
                     ' dictionary')
        raise GMOS_IMAGEException('The CLscript '+CLscript+
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
            'Stdout'    :gemt.IrafStdout(verbose=verbose),
            'Stderr'    :gemt.IrafStdout(verbose=verbose)
                       }
        return defaultParams
