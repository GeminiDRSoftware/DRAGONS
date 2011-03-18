# Author: Kyle Mede. 2010
# Skeleton originally written by Craig Allen, callen@gemini.edu
import sys, StringIO, os

from astrodata import Descriptors
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from astrodata.ConfigSpace import lookupPath
from astrodata.data import AstroData
from astrodata.Errors import PrimitiveError
from gempy.instruments import geminiTools as gemt
from gempy.instruments import gmosTools as gmost
from gempy.science import geminiScience
from gempy.science import gmosScience
from primitives_GEMINI import GEMINIPrimitives
import shutil

class GMOSPrimitives(GEMINIPrimitives):
    """ 
    This is the class of all primitives for the GMOS level of the type 
    hierarchy tree.  It inherits all the primitives to the level above
    , 'GEMINIPrimitives'.
    
    """
    astrotype = 'GMOS'
    
    def init(self, rc):
        GEMINIPrimitives.init(self, rc)
        return rc
     
    def addBPM(self,rc):
        """
        This primitive is used by the general addDQ primitive of 
        primitives_GEMINI to add the appropriate BPM (Bad Pixel Mask)
        to the inputs.  This function will add the BPM as frames matching
        that of the SCI frames and ensure the BPM's data array is the same 
        size as that of the SCI data array. If the SCI array is larger 
        (say SCI's were overscan trimmed, but BPMs were not), the BPMs will 
        have their arrays padded with zero's to match the sizes and use the 
        data_section descriptor on the SCI data arrays to ensure the match is
        a correct fit.
        
        Using this approach, rather than appending the BPM in the addDQ allows
        for specialized BPM processing to be done in the instrument specific
        primitive sets where it belongs.                          
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logName=rc['logName'], logLevel=rc['logLevel'])
        try:
            log.status('*STARTING* to add the BPM frame(s) to the input data')
            
            #$$$$$$$$$$$$$ TO BE callibration search, correct when ready $$$$$$$
            BPM_11 = AstroData(lookupPath('Gemini/GMOS/GMOS_BPM_11.fits'))
            BPM_22 = AstroData(lookupPath('Gemini/GMOS/GMOS_BPM_22.fits'))
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            
            # Instantiate a list of suitable BPMs to be passed to addBPM func 
            BPMlist = []
            
            # Loop through inputs and load up BPMlist
            for ad in rc.getInputs(style='AD'):
                ### This section might need to be upgraded in the future for more 
                ### general use instead of just 1x1 and 2x2 imaging
                if ad[('SCI',1)].getKeyValue('CCDSUM')=='1 1':
                    BPMlist.append(BPM_11)
                elif ad[('SCI',1)].getKeyValue('CCDSUM')=='2 2':
                    BPMlist.append(BPM_22)
                else:
                    log.error('CCDSUM is not 1x1 or 2x2')
                    #$$$ NOT REALLY SURE THIS IS THE APPROPRIATE ACTION HERE
                    raise
   
            log.debug('Calling geminiScience.addBPM function')
            
            adOutputs = geminiScience.add_bpm(adInputs=rc.getInputs(style='AD'), 
                                         BPMs=BPMlist,matchSize=True, suffix=rc['suffix'], 
                                         log=log)           
            
            log.status('geminiScience.addBPM completed successfully')
                
            # Reporting the updated files to the reduction context
            rc.reportOutput(adOutputs)   
                
            log.status('*FINISHED* adding the BPM to the inputs') 
        except:
            # logging the exact message from the actual exception that was 
            # raised in the try block. Then raising a general PrimitiveError 
            # with message.
            log.critical(repr(sys.exc_info()[1]))
            raise PrimitiveError('Problem processing one of '+rc.inputsAsStr())
            
        yield rc       
        
    def biasCorrect(self, rc):
        """
        This primitive will subtract the biases from the inputs using the 
        CL script gireduce.
        
        WARNING: The gireduce script used here replaces the previously 
        calculated DQ frames with its own versions.  This may be corrected 
        in the future by replacing the use of the gireduce
        with a Python routine to do the bias subtraction.
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param fl_over: Subtract the overscan?
        :type fl_over: Python boolean (True/False), default is False
        
        :param fl_trim: Trim the overscan region from the frames?
        :type fl_trim: Python boolean (True/False), default is False. 
                       Note: This value cannot be set during a recipe or from 
                       reduce command line call, only in the parameter file.
        
        :param fl_vardq: Create variance and data quality frames?
        :type fl_vardq: Python boolean (True/False)
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
#        # Loading and bringing the pyraf related modules into the name-space
#        pyraf, gemini, yes, no = pyrafLoader()
        
        log = gemLog.getGeminiLog(logName=rc['logName'], logLevel=rc['logLevel'])
        try:
            log.status('*STARTING* to subtract the bias from the inputs')
            
            # Getting the bias file for the first file of the inputs and 
            # assuming it is the same for all the inputs. This should be 
            # corrected in the future to be more intelligent and get the 
            # correct bias for each input individually if they are not 
            # all the same. Then gireduce can be called in a loop with 
            # one flat and one bias, this will work well with the CLManager
            # as that was how i wrote this prim originally.
            adOne = rc.getInputs(style='AD')[0]
            processedBias = AstroData(rc.getCal(adOne,'bias'))
            ####################BULL CRAP FOR TESTING ########################## 
            #from copy import deepcopy
            #processedBias = deepcopy(adOne)
            #processedBias.filename = 'TEMPNAMEforBIAS.fits'
            #processedBias.phuSetKeyValue('ORIGNAME','TEMPNAMEforBIAS.fits')
            #processedBias.historyMark(key='GBIAS', 
            #                  comment='fake key to trick CL that GBIAS was ran')
            ####################################################################
            log.status('Using bias '+processedBias.filename+' to correct the inputs')
            log.debug('Calling geminiScience.biasCorrect function')
            
            adOutputs = geminiScience.bias_correct(adInputs=rc.getInputs(style='AD'), 
                                         biases=processedBias, fl_vardq=rc['fl_vardq'], 
                                         fl_trim=rc['fl_trim'], fl_over=rc['fl_over'], 
                                         suffix=rc['suffix'], log=log)           
            
            log.status('geminiScience.biasCorrect completed successfully')
                
            # Reporting the updated files to the reduction context
            rc.reportOutput(adOutputs)   
            
            log.status('*FINISHED* subtracting the bias from the input flats')
        except:
            # logging the exact message from the actual exception that was 
            # raised in the try block. Then raising a general PrimitiveError 
            # with message.
            log.critical(repr(sys.exc_info()[1]))
            raise PrimitiveError('Problem processing one of '+rc.inputsAsStr())
            
        yield rc

    def display(self, rc):
        """ 
        This is a primitive for displaying GMOS data.
        It utilizes the IRAF routine gdisplay and requires DS9 to be running
        before this primitive is called.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logName=rc['logName'], logLevel=rc['logLevel'])
        try:
            #from astrodata.adutils.future import gemDisplay
            #ds = gemDisplay.getDisplayService()
            
            log.status('*STARTING* to display the images of the input data')
            
            # Loading and bringing the pyraf related modules into the name-space
            pyraf, gemini, yes, no = pyrafLoader()
            
            # Ensuring image buffer is large enough to handle GMOS images
            pyraf.iraf.set(stdimage='imtgmos')              
                
            for i in range(0, len(rc.inputs)):  
                # Retrieving the input object for this increment from the RC 
                inputRecord = rc.inputs[i]
                
                # Creating a dictionary of the parameters set by definition of the primitive 
                clPrimParams = {
                'image'         :inputRecord.filename,
                # Using the increment value (+1) for the frame value
                'frame'         :i+1,
                'fl_imexam'     :no,
                # Retrieving the observatory key from the PHU
                'observatory'   :inputRecord.ad.phuGetKeyValue('OBSERVAT')
                                }
                
                # Grabbing the default parameters dictionary and updating 
                # it with the above dictionary
                clParamsDict = CLDefaultParamsDict('gdisplay',logLevel=rc['logLevel'])
                clParamsDict.update(clPrimParams)
                
                # Logging the values in the prim parameter dictionaries
                log.fullinfo('\nParameters dictated by the definition of the '+
                         'primitive:\n', 
                         category='parameters')
                gemt.LogDictParams(clPrimParams)
                
                log.debug('Calling the gdisplay CL script for input list '+
                              inputRecord.filename)
                
                try:
                    gemini.gmos.gdisplay(**clParamsDict)
                    
                    if gemini.gmos.gdisplay.status:
                        raise PrimitiveError('gdisplay failed for input '+
                                     inputRecord.filename)
                    else:
                        log.status('Exited the gdisplay CL script successfully')
                        
                except:
                    # This exception should allow for a smooth exiting if there is an 
                    # error with gdisplay, most likely due to DS9 not running yet
                    log.error('ERROR occurred while trying to display '+str(inputRecord.filename)
                                        +', ensure that DS9 is running and try again')
                    
                # this version had the display id conversion code which we'll need to redo
                # code above just uses the loop index as frame number
                #gemini.gmos.gdisplay( inputRecord.filename, ds.displayID2frame(rq.disID), fl_imexam=iraf.no,
                #    Stdout = coi.getIrafStdout(), Stderr = coi.getIrafStderr() )
                
            log.status('*FINISHED* displaying the images of the input data')
        except:
            # logging the exact message from the actual exception that was 
            # raised in the try block. Then raising a general PrimitiveError 
            # with message.
            log.critical(repr(sys.exc_info()[1]))
            raise PrimitiveError('There was a problem displaying '+
                                 rc.inputsAsStr())
        yield rc

    def localGetProcessedBias(self,rc):
        """
        A prim that works with the calibration system (MAYBE), but as it isn't 
        written yet this simply copies the bias file from the stored processed 
        bias directory and reports its name to the reduction context. 
        This is the basic form that the calibration system will work as well 
        but with proper checking for what the correct bias file would be rather 
        than my oversimplified checking the bining alone.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logName=rc['logName'], logLevel=rc['logLevel'])
        try:
            packagePath = sys.argv[0].split('gemini_python')[0]
            calPath = 'gemini_python/test_data/test_cal_files/processed_biases/'
            
            for ad in rc.getInputs(style='AD'):
                if ad.extGetKeyValue(1,'CCDSUM') == '1 1':
                    log.error('NO 1x1 PROCESSED BIAS YET TO USE')
                    raise 'error'
                elif ad.extGetKeyValue(1,'CCDSUM') == '2 2':
                    biasfilename = 'N20020214S022_preparedBias.fits'
                    if not os.path.exists(os.path.join('.reducecache/'+
                                                       'storedcals/retrievd'+
                                                       'biases', biasfilename)):
                        shutil.copy(packagePath+calPath+biasfilename, 
                                    '.reducecache/storedcals/retrievedbiases')
                    rc.addCal(ad,'bias', 
                              os.path.join('.reducecache/storedcals/retrieve'+
                                           'dbiases',biasfilename))
                else:
                    log.error('CCDSUM is not 1x1 or 2x2 for the input flat!!')
           
        except:
            # logging the exact message from the actual exception that was 
            # raised in the try block. Then raising a general PrimitiveError 
            # with message.
            log.critical(repr(sys.exc_info()[1]))
            raise PrimitiveError('Problem preparing one of '+rc.inputsAsStr())
        yield rc
   
    def localGetProcessedFlat(self,rc):
        """
        A prim that works with the calibration system (MAYBE), but as it 
        isn't written yet this simply copies the bias file from the stored 
        processed bias directory and reports its name to the reduction 
        context. this is the basic form that the calibration system will work 
        as well but with proper checking for what the correct bias file would 
        be rather than my oversimplified checking
        the binning alone.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logName=rc['logName'], logLevel=rc['logLevel'])
        try:
            packagePath=sys.argv[0].split('gemini_python')[0]
            calPath='gemini_python/test_data/test_cal_files/processed_flats/'
            
            for ad in rc.getInputs(style='AD'):
                if ad.extGetKeyValue(1,'CCDSUM') == '1 1':
                    log.error('NO 1x1 PROCESSED BIAS YET TO USE')
                    raise 'error'
                elif ad.extGetKeyValue(1,'CCDSUM') == '2 2':
                    flatfilename = 'N20020211S156_preparedFlat.fits'
                    if not os.path.exists(os.path.join('.reducecache/storedca'+
                                                       'ls/retrievedflats', 
                                                       flatfilename)):
                        shutil.copy(packagePath+calPath+flatfilename, 
                                    '.reducecache/storedcals/retrievedflats')
                    rc.addCal(ad,'flat', os.path.join('.reducecache/storedca'+
                                                      'ls/retrievedflats', 
                                                      flatfilename))
                else:
                    log.error('CCDSUM is not 1x1 or 2x2 for the input image!!')
           
        except:
            # logging the exact message from the actual exception that was 
            # raised in the try block. Then raising a general PrimitiveError 
            # with message.
            log.critical(repr(sys.exc_info()[1]))
            raise PrimitiveError('Problem retrieving one of '+rc.inputsAsStr())
        
        yield rc

    def mosaicDetectors(self,rc):
        """
        This primitive will mosaic the SCI frames of the input images, 
        along with the VAR and DQ frames if they exist.  
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param fl_paste: Paste images instead of mosaic?
        :type fl_paste: Python boolean (True/False), default is False
        
        :param interp_function: Type of interpolation function to use accross the chip gaps
        :type interp_function: string, options: 'linear', 'nearest', 'poly3', 
                               'poly5', 'spine3', 'sinc'.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logName=rc['logName'], logLevel=rc['logLevel'])
        
        # loading and bringing the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader()
        
        try:
            log.status('*STARTING* to mosaic the input images SCI extensions'+
                       ' together')
            
            log.debug('Calling geminiScience.mosaicDetectors function')
            
            adOutputs = geminiScience.mosaic_detectors(adInputs=rc.getInputs(style='AD'), 
                                        fl_paste=rc['fl_paste'], interp_function=rc['interp_function'], 
                                        fl_vardq='AUTO', suffix=rc['suffix'], 
                                        log=log)           
            
            log.status('geminiScience.mosaicDetectors completed successfully')
                
            # Reporting the updated files to the reduction context
            rc.reportOutput(adOutputs) 
                
            log.status('*FINISHED* mosaicing the input images')
        except:
            # logging the exact message from the actual exception that was 
            # raised in the try block. Then raising a general PrimitiveError 
            # with message.
            log.critical(repr(sys.exc_info()[1]))
            raise PrimitiveError('Problem processing one of '+rc.inputsAsStr())
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
        log = gemLog.getGeminiLog(logName=rc['logName'], logLevel=rc['logLevel'])
        
        # Loading and bringing the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader()
        
        try:
            log.status('*STARTING* to combine and normalize the input flats')
            
            log.debug('Calling geminiScience.normalizeFlat function')
            
            adOutputs = geminiScience.normalize_flat(adInputs=rc.getInputs(style='AD'), 
                                        fl_trim=rc['fl_trim'], fl_over=rc['fl_over'], 
                                        fl_vardq='AUTO', suffix=rc['suffix'], 
                                        log=log)           
            
            log.status('geminiScience.normalizeFlat completed successfully')
                
            # Reporting the updated files to the reduction context
            rc.reportOutput(adOutputs)
        
            log.status('*FINISHED* combining and normalizing the input flats')
        except:
            # logging the exact message from the actual exception that was 
            # raised in the try block. Then raising a general PrimitiveError 
            # with message.
            log.critical(repr(sys.exc_info()[1]))
            raise PrimitiveError('Problem processing one of '+rc.inputsAsStr())
            
        yield rc

    def overscanSubtract(self,rc):
        """
        This primitive uses the CL script gireduce to subtract the overscan 
        from the input images.
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param fl_trim: Trim the overscan region from the frames?
        :type fl_trim: Python boolean (True/False)
        
        :param fl_vardq: Create variance and data quality frames?
        :type fl_vardq: Python boolean (True/False)
            
        :param biassec: biassec parameter of format '[#:#,#:#],[#:#,#:#],[#:#,#:#]'
        :type biassec: string. default: '[1:25,1:2304],[1:32,1:2304],[1025:1056,1:2304]' is ideal for 2x2 GMOS data.
    
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logName=rc['logName'], logLevel=rc['logLevel'])
        # Loading and bringing the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader()
        
        try:
            log.status('*STARTING* to subtract the overscan from the inputs')
            
            log.debug('Calling gmosScience.overscanSubtract function')
            
            adOutputs = gmosScience.overscan_subtract(adInputs=rc.getInputs(style='AD'), 
                                        fl_trim=rc['fl_trim'], biassec=rc['biassec'], 
                                        fl_vardq='AUTO', suffix=rc['suffix'], 
                                        log=log)           
            
            log.status('gmosScience.overscanSubtract completed successfully')
                
            # Reporting the updated files to the reduction context
            rc.reportOutput(adOutputs)
            
            log.status('*FINISHED* subtracting the overscan from the '+
                       'input data')
        except:
            # logging the exact message from the actual exception that was 
            # raised in the try block. Then raising a general PrimitiveError 
            # with message.
            log.critical(repr(sys.exc_info()[1]))
            raise PrimitiveError('Problem processing one of '+rc.inputsAsStr())
        
        yield rc    

    def overscanTrim(self,rc):
        """
        This primitive uses AstroData to trim the overscan region 
        from the input images and update their headers.
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logName=rc['logName'], logLevel=rc['logLevel'])
        try:
            log.status('*STARTING* to trim the overscan region from the input data')
            
            log.debug('Calling geminiScience.overscanTrim function')
            
            adOutputs = geminiScience.overscan_trim(adInputs=rc.getInputs(style='AD'),     
                                                        suffix=rc['suffix'], 
                                                        log=log)           
            
            log.status('geminiScience.overscanTrim completed successfully')
              
            # Reporting the updated files to the reduction context
            rc.reportOutput(adOutputs)   
                
            log.status('*FINISHED* trimming the overscan region from the input data')
        except:
            # logging the exact message from the actual exception that was 
            # raised in the try block. Then raising a general PrimitiveError 
            # with message.
            log.critical(repr(sys.exc_info()[1]))
            raise PrimitiveError('Problem processing one of '+rc.inputsAsStr())
        
        yield rc
         
    def standardizeInstrumentHeaders(self,rc):
        """
        This primitive is called by standardizeHeaders to makes the changes and 
        additions to the headers of the input files that are instrument 
        specific.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logName=rc['logName'], logLevel=rc['logLevel'])
        try:                                          
            for ad in rc.getInputs(style='AD'): 
                log.debug('Calling gmost.stdInstHdrs for '+ad.filename) 
                gmost.stdInstHdrs(ad, logLevel=rc['logLevel']) 
                log.status('Completed standardizing instrument headers for '+
                           ad.filename)
                    
        except:
            # logging the exact message from the actual exception that was 
            # raised in the try block. Then raising a general PrimitiveError 
            # with message.
            log.critical(repr(sys.exc_info()[1]))
            raise PrimitiveError('Problem preparing one of '+rc.inputsAsStr())
        
        yield rc 
    
    def validateInstrumentData(self,rc):
        """
        This primitive is called by validateData to validate the instrument 
        specific data checks for all input files.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logName=rc['logName'], logLevel=rc['logLevel'])
        try:
            for ad in rc.getInputs(style='AD'):
                log.debug('Calling gmost.valInstData for '+ad.filename)
                gmost.valInstData(ad)
                log.status('Completed validating instrument data for '+
                           ad.filename)
                
        except:
            # logging the exact message from the actual exception that was 
            # raised in the try block. Then raising a general PrimitiveError 
            # with message.
            log.critical(repr(sys.exc_info()[1]))
            raise PrimitiveError('Problem preparing one of '+rc.inputsAsStr())
        
        yield rc

