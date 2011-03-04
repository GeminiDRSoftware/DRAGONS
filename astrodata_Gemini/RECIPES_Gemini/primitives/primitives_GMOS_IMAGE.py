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
from gempy.science import gmosScience
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
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param fl_statscale: Scale by statistics rather than exposure time
        :type fl_statscale: Boolean
        
        :param statsec: image section used to determine the scale factor 
                        if fl_statsec=True
        :type statsec: string of format '[EXTNAME,EXTVER][x1:x2,y1:y2]'
                       default: If CCDSUM = '1 1' :[SCI,2][100:1900,100:4500]'
                       If CCDSUM = '2 2' : [SCI,2][100:950,100:2250]'
        
        :param scale: Override auto-scaling if not 0.0
        :type scale: real
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        log = gemLog.getGeminiLog(logLevel=rc['logLevel'])
        try: 
            log.status('*STARTING* to fringe correct the images')
            
            # Using the same fringe file for all the input images
            #fringe = $$$$$$$$$$$$$$$$$$$$$$$$$$$ ??? where to get it  ?????
            #$$$$$$$ TEMP $$$$$$$$$$$$
            fringe = rc.getInputs(style='AD')[0]
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            
            log.debug('Calling gmosScience.fringe_correct function')
            
            adOuts = gmosScience.fringe_correct(adIns=rc.getInputs(style='AD'), 
                                         fringes=fringe,
                                         fl_statscale=rc['fl_statscale'],
                                         statsec=rc['statsec'], scale=rc['scale'],
                                         suffix=rc['suffix'], 
                                         logLevel=rc['logLevel'])           
            
            log.status('gmosScience.fringe_correct completed successfully')
                
            # Reporting the updated files to the reduction context
            rc.reportOutput(adOuts)              
                
            log.status('*FINISHED* fringe correcting the input data')
            
        except:
            log.critical("Problem subtracting fringe from "+rc.inputsAsStr())
            raise 

        yield rc 
    
    def makeFringeFrame(self, rc):
        """
        This primitive will create a single fringe image from all the inputs.
        It utilizes the CL script gifringe to create the fringe image.
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param fl_vardq: Create variance and data quality frames?
        :type fl_vardq: Python boolean (True/False)
        
        :param method: type of combining method to use on the fringe frames.
        :type method: string, options: 'average', 'median
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        """
        log = gemLog.getGeminiLog(logLevel=rc['logLevel'])        
        try:
            if len(rc.getInputs())>1:
                log.status('*STARTING* to create a fringe frame from the inputs')
    
                log.debug('Calling gmosScience.make_fringe_frame_imaging function')
            
                adOuts = gmosScience.make_fringe_frame_imaging(adIns=rc.getInputs(style='AD'), 
                                             fl_vardq=rc['fl_vardq'],
                                             method=rc['method'],
                                             suffix=rc['suffix'], 
                                             logLevel=rc['logLevel'])           
                
                log.status('gmosScience.make_fringe_frame_imaging completed successfully')
                    
                # Reporting the updated files to the reduction context
                rc.reportOutput(adOuts)              
                
            else:
                log.status('makeFringeFrame was called with only one input, '+\
                           'so it just passed the inputs through without doing'+\
                           ' anything to them.')
            # Reporting the updated files to the reduction context
            rc.reportOutput(rc.getInputs(style='AD'))
            
            log.status('*FINISHED* creating the fringe image')
        except:
            log.critical("Problem creating fringe from "+rc.inputsAsStr())
            raise 
        yield rc  

def CLDefaultParamsDict(CLscript, logLevel=1):
    """
    A function to return a dictionary full of all the default parameters 
    for each CL script used so far in the Recipe System.
    
    :param logLevel: Verbosity setting for log messages to the screen.
    :type logLevel: int. Default: 1
    """
    log = gemLog.getGeminiLog(logLevel=logLevel)
    
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
            'Stdout'    :gemt.IrafStdout(logLevel=logLevel),
            'Stderr'    :gemt.IrafStdout(logLevel=logLevel)
                       }
        return defaultParams
