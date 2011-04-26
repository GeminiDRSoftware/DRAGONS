# Author: Kyle Mede. 2010
# Skeleton originally written by Craig Allen, callen@gemini.edu
import sys, StringIO, os

import time
from astrodata.adutils import filesystem
from astrodata.adutils import gemLog
from astrodata import IDFactory
from astrodata import Descriptors
from astrodata.data import AstroData
from astrodata.Errors import PrimitiveError
from primitives_GEMINI import GEMINIPrimitives
from primitives_GEMINI import pyrafLoader
from primitives_GMOS import GMOSPrimitives
from gempy.science import calibrate
from gempy.science import gmosScience
from gempy.science import standardize
from gempy import geminiTools as gemt
from gempy.geminiCLParDicts import CLDefaultParamsDict

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
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logType=rc['logType'],logLevel=rc['logLevel'])        
        try:
            if len(rc.getInputs())>1:
                log.status('*STARTING* to create a fringe frame from the inputs')
    
                log.debug('Calling gmosScience.make_fringe_frame_imaging function')
            
                adOutputs = gmosScience.make_fringe_frame_imaging(
                                                adInputs=rc.getInputs(style='AD'), 
                                             fl_vardq=rc['fl_vardq'],
                                             method=rc['method'],
                                             suffix=rc['suffix'])           
                
                log.status('gmosScience.make_fringe_frame_imaging completed'+
                           ' successfully')
                    
                # Reporting the updated files to the reduction context
                rc.reportOutput(adOutputs)              
                
            else:
                log.status('makeFringeFrame was called with only one input, '+
                           'so it just passed the inputs through without '+
                           'doing anything to them.')
            # Reporting the updated files to the reduction context
            rc.reportOutput(rc.getInputs(style='AD'))
            
            log.status('*FINISHED* creating the fringe image')
        except:
            # logging the exact message from the actual exception that was 
            # raised in the try block. Then raising a general PrimitiveError 
            # with message.
            log.critical(repr(sys.exc_info()[1]))
            raise 
        yield rc  
        
    def normalizeFlat(self, rc):
        """
        This primitive will combine the input flats and then normalize them 
        using the CL script giflat.
        
        Warning: giflat calculates its own DQ frames and thus replaces the 
        previously produced ones in calculateDQ. This may be fixed in the 
        future by replacing giflat with a Python equivilent with more 
        appropriate options for the recipe system.
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param fl_over: Subtract the overscan level from the frames?
        :type fl_over: Python boolean (True/False)
        
        :param fl_trim: Trim the overscan region from the frames?
        :type fl_trim: Python boolean (True/False)
        
        :param fl_vardq: Create variance and data quality frames?
        :type fl_vardq: Python boolean (True/False)
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logType=rc['logType'],logLevel=rc['logLevel'])
        
        # Loading and bringing the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader()
        
        try:
            log.status('*STARTING* to combine and normalize the input flats')
            
            log.debug('Calling calibrate.normalize_flat_image_gmos function')
            
            adOutputs = calibrate.normalize_flat_image_gmos(
                                            adInputs=rc.getInputs(style='AD'), 
                                        fl_trim=rc['fl_trim'], 
                                        fl_over=rc['fl_over'], 
                                        fl_vardq='AUTO', suffix=rc['suffix'])           
            
            log.status('calibrate.normalize_flat_image_gmos completed '+
                                                                'successfully')
                
            # Reporting the updated files to the reduction context
            rc.reportOutput(adOutputs)
        
            log.status('*FINISHED* combining and normalizing the input flats')
        except:
            # logging the exact message from the actual exception that was 
            # raised in the try block. Then raising a general PrimitiveError 
            # with message.
            log.critical(repr(sys.exc_info()[1]))
            raise 
        yield rc
        
    def standardizeStructure(self,rc):
        """
        This primitive will to add an MDF to the
        inputs if they are of type SPECT, those of type IMAGE will be handled
        by the standardizeStructure in the primitives_GMOS_IMAGE set
        where no MDF will be added.
        The Science Function standardize_structure_gmos in standardize.py is
        utilized to do the work for this primitive.
        
        :param suffix: Value to be post pended onto each input name(s) to 
                       create the output name(s).
        :type suffix: string
        
        :param addMDF: A flag to turn on/off appending the appropriate MDF 
                       file to the inputs.
        :type addMDF: Python boolean (True/False)
                      default: True
                      
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logType=rc['logType'], logLevel=rc['logLevel'])
        try:           
            log.status('*STARTING* to standardize the structure (GMOS_IMAGE)')                         
            log.debug('Calling standardize.standardize_structure_gmos')
            adOutputs = standardize.standardize_structure_gmos(
                                            adInputs=rc.getInputs(style='AD'),
                                                        addMDF=rc['addMDF'],     
                                                        suffix=rc['suffix'])
            
            # Reporting the updated files to the reduction context
            rc.reportOutput(adOutputs) 
            
            log.status('*FINISHED* standardizing the structure (GMOS_IMAGE)')        
        except:
            # logging the exact message from the actual exception that was 
            # raised in the try block. Then raising a general PrimitiveError 
            # with message.
            log.critical(repr(sys.exc_info()[1]))
            raise 
        yield rc     


