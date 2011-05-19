# Author: Kyle Mede. 2010
# Skeleton originally written by Craig Allen, callen@gemini.edu

import os, shutil, sys
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from astrodata.ConfigSpace import lookup_path
from astrodata.data import AstroData
from astrodata import Errors
from gempy import geminiTools as gt
from gempy.science import geminiScience as gs
from gempy.science import preprocessing as pp
from gempy.science import standardization as sdz
from gempy.geminiCLParDicts import CLDefaultParamsDict
from primitives_GEMINI import GEMINIPrimitives

class GMOSPrimitives(GEMINIPrimitives):
    """
    This is the class containing all of the primitives for the GMOS level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'GEMINIPrimitives'.
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

        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logType=rc['logType'],
                                  logLevel=rc['logLevel'])
        log.debug(gt.log_message('primitive', 'addBPM', 'starting'))
            
        adoutput_list = []

        #$$$$$$$$$$$$$ TO BE callibration search, correct when ready $$$$$$$
        BPM_11 = AstroData(lookup_path('Gemini/GMOS/BPM/GMOS_BPM_11.fits'))
        BPM_22 = AstroData(lookup_path('Gemini/GMOS/BPM/GMOS_BPM_22.fits'))
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            
        # Loop through inputs, adding appropriate mask
        for ad in rc.get_inputs(style='AD'):
            if ad.phu_get_key_value('ADDBPM'):
                log.warning('%s has already been processed by addBPM' %
                            (ad.filename))
                adoutput_list.append(ad)
                continue

            ### This section might need to be upgraded in the future
            ### for more general use instead of just 1x1 and 2x2 imaging
            if ad[('SCI',1)].get_key_value('CCDSUM')=='1 1':
                bpm = BPM_11
            elif ad[('SCI',1)].get_key_value('CCDSUM')=='2 2':
                bpm = BPM_22
            else:
                log.error('CCDSUM is not 1x1 or 2x2')
                #$$$ NOT REALLY SURE THIS IS THE APPROPRIATE ACTION HERE
                raise
   
            ad = gs.add_bpm(adinput=ad,bpm=bpm)
            adoutput_list.append(ad[0])
            
        # Report the updated files to the reduction context
        rc.report_output(adoutput_list)   
                
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
        log = gemLog.getGeminiLog(logType=rc['logType'],
                                  logLevel=rc['logLevel'])
        log.debug(gt.log_message('primitive', 'display', 'starting'))

        
        # Loading and bringing the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader()
            
        # Ensuring image buffer is large enough to handle GMOS images
        pyraf.iraf.set(stdimage='imtgmos')              
                
        for i in range(0, len(rc.inputs)):  
            # Retrieving the input object for this increment from the RC 
            inputRecord = rc.inputs[i]
                
            # Creating a dictionary of the parameters set by 
            # definition of the primitive 
            clPrimParams = {
                'image'         :inputRecord.filename,
                # Using the increment value (+1) for the frame value
                'frame'         :i+1,
                'fl_imexam'     :no,
                # Retrieving the observatory key from the PHU
                'observatory'   :inputRecord.ad.phu_get_key_value('OBSERVAT')
                }
                
            # Grabbing the default parameters dictionary and updating 
            # it with the above dictionary
            clParamsDict = CLDefaultParamsDict('gdisplay')
            clParamsDict.update(clPrimParams)
                
            # Logging the values in the prim parameter dictionaries
            log.fullinfo('\nParameters dictated by the definition of the '+
                         'primitive:\n', 
                         category='parameters')
            gt.logDictParams(clPrimParams)
                
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
                log.error('ERROR occurred while trying to display '+
                          str(inputRecord.filename)
                          +', ensure that DS9 is running and try again')
                    
            # this version had the display id conversion code which we'll need to redo
            # code above just uses the loop index as frame number
            #gemini.gmos.gdisplay( inputRecord.filename, ds.displayID2frame(rq.dis_id),
            #                      fl_imexam=iraf.no,
            #    Stdout = coi.get_iraf_stdout(), Stderr = coi.get_iraf_stderr() )
                
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
        log = gemLog.getGeminiLog(logType=rc['logType'],
                                  logLevel=rc['logLevel'])
        log.debug(gt.log_message('primitive', 'localGetProcessedBias', 'starting'))

        packagePath = sys.argv[0].split('gemini_python')[0]
        calPath = 'gemini_python/test_data/test_cal_files/processed_biases/'
            
        for ad in rc.get_inputs(style='AD'):
            if ad.ext_get_key_value(1,'CCDSUM') == '1 1':
                log.error('NO 1x1 PROCESSED BIAS YET TO USE')
                raise 'error'
            elif ad.ext_get_key_value(1,'CCDSUM') == '2 2':
                biasfilename = 'N20020214S022_preparedBias.fits'
                if not os.path.exists(os.path.join('.reducecache/'+
                                                   'storedcals/retrieved'+
                                                   'biases', biasfilename)):
                    shutil.copy(packagePath+calPath+biasfilename, 
                                '.reducecache/storedcals/retrievedbiases')
                rc.add_cal(ad,'bias', 
                           os.path.join('.reducecache/storedcals/retrieve'+
                                        'dbiases',biasfilename))
            else:
                log.error('CCDSUM is not 1x1 or 2x2 for the input flat')
           
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
        log = gemLog.getGeminiLog(logType=rc['logType'],logLevel=rc['logLevel'])
        log.debug(gt.log_message('primitive', 'localGetProcessedFlat', 'starting'))

        packagePath=sys.argv[0].split('gemini_python')[0]
        calPath='gemini_python/test_data/test_cal_files/processed_flats/'
            
        for ad in rc.get_inputs(style='AD'):
            if ad.ext_get_key_value(1,'CCDSUM') == '1 1':
                log.error('NO 1x1 PROCESSED BIAS YET TO USE')
                raise 'error'
            elif ad.ext_get_key_value(1,'CCDSUM') == '2 2':
                flatfilename = 'N20020211S156_preparedFlat.fits'
                if not os.path.exists(os.path.join('.reducecache/storedcals/'+
                                                   'retrievedflats', 
                                                   flatfilename)):
                    shutil.copy(packagePath+calPath+flatfilename, 
                                '.reducecache/storedcals/retrievedflats')
                rc.add_cal(ad,'flat', os.path.join('.reducecache/storedcals/'+
                                                   'retrievedflats', 
                                                   flatfilename))
            else:
                log.error('CCDSUM is not 1x1 or 2x2 for the input image!!')

        yield rc

    def mosaicDetectors(self,rc):
        """
        This primitive will mosaic the SCI frames of the input images, 
        along with the VAR and DQ frames if they exist.  
        
        :param tile: tile images instead of mosaic
        :type tile: Python boolean (True/False), default is False
        
        :param interpolator: Type of interpolation function to use accross
                             the chip gaps
        :type interpolator: string, options: 'linear', 'nearest', 'poly3', 
                            'poly5', 'spine3', 'sinc'.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logType=rc['logType'],
                                  logLevel=rc['logLevel'])
        log.debug(gt.log_message('primitive', 'mosaicDetectors', 'starting'))
        
        adoutput_list = []
        for ad in rc.get_inputs(style='AD'):
            if ad.phu_get_key_value('MOSAIC'):
                log.warning('%s has already been processed by mosaicDetectors' %
                            (ad.filename))
                adoutput_list.append(ad)
                continue
                
            ad = gs.mosaic_detectors(adinput=ad, 
                                     tile=rc['tile'], 
                                     interpolator=rc['interpolator'])           
            
            adoutput_list.append(ad[0])
                
        rc.report_output(adoutput_list)                
        yield rc

    def overscanSubtract(self,rc):
        """
        This primitive uses the CL script gireduce to subtract the overscan 
        from the input images.
        
        :param trim: Trim the overscan region from the frames?
        :type trim: Python boolean (True/False)
        
        :param overscan_section: biassec parameter of format 
                                 '[#:#,#:#],[#:#,#:#],[#:#,#:#]'
        :type overscan_section: string. default: 
                                '[1:25,1:2304],[1:32,1:2304],[1025:1056,1:2304]' 
                                is ideal for 2x2 GMOS data.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logType=rc['logType'],
                                  logLevel=rc['logLevel'])
        log.debug(gt.log_message('primitive', 'overscanSubtract', 'starting'))
        
        adoutput_list = []
        for ad in rc.get_inputs(style='AD'):
            if ad.phu_get_key_value('OVERSUB'):
                log.warning('%s has already been processed by overscanSubtract' %
                            (ad.filename))
                adoutput_list.append(ad)
                continue

            ad = pp.overscan_subtract_gmosNEW(adinput=ad, trim=rc['trim'], 
                                              overscan_section=rc['overscan_section'])
            adoutput_list.append(ad[0])

        rc.report_output(adOutputs)

        yield rc    

    def overscanTrim(self,rc):
        """
        This primitive uses AstroData to trim the overscan region 
        from the input images and update their headers.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logType=rc['logType'],
                                  logLevel=rc['logLevel'])

        log.debug(gt.log_message('primitive', 'overscanTrim', 'starting'))
        
        adoutput_list = []
        for ad in rc.get_inputs(style='AD'):
            if ad.phu_get_key_value('OVERTRIM'):
                log.warning('%s has already been processed by overscanTrim' %
                            (ad.filename))
                adoutput_list.append(ad)
                continue
            
            ad = pp.overscan_trim(adinput=ad)
            adoutput_list.append(ad[0])

        rc.report_output(adOutputs)   

        yield rc
         
    def standardizeHeaders(self,rc):
        """
        This primitive is used to update and add keywords to the headers of the
        input dataset. First, it calls the standardize_headers_gemini user
        level function to update Gemini specific keywords and then updates GMOS
        specific keywords.
        
        Either a 'main' type logger object, if it exists, or a null logger
        (i.e., no log file, no messages to screen) will be retrieved
        and used within this function.
          
        :param loglevel: Verbosity setting for log messages to the screen.
                         0 = nothing to screen, 6 = everything to screen. OR
                         the message level as a string (i.e., 'critical',
                         'status', 'fullinfo' ...)
        :type loglevel: integer or string
        """
        log = gemLog.getGeminiLog(logType=rc['logType'],
                                  logLevel=rc['logLevel'])
        log.debug(gt.log_message('primitive', 'standardizeHeaders', 'starting'))

        adoutput_list = []
        for ad in rc.get_inputs(style='AD'):
            if ad.phu_get_key_value('STDHDRSI'):
                log.warning('%s has already been processed by standardizeHeaders' %
                            (ad.filename))
                adoutput_list.append(ad)
                continue
 
            ad = sdz.standardize_headers_gmos(adinput=ad)
            adoutput_list.append(ad[0])

        rc.report_output(output)
        yield rc

    def standardizeStructure(self,rc):
        """
        This primitive will to add an MDF to the
        inputs if they are of type SPECT, those of type IMAGE will be handled
        by the standardizeStructure in the primitives_GMOS_IMAGE set
        where no MDF will be added.
        The Science Function standardize_structure_gmos in standardize.py is
        utilized to do the work for this primitive.
        
        :param add_mdf: A flag to turn on/off appending the appropriate MDF 
                       file to the inputs.
        :type add_mdf: Python boolean (True/False)
                      default: True
                      
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logType=rc['logType'], 
                                  logLevel=rc['logLevel'])
        log.debug(gt.log_message('primitive', 'standardizeStructure', 'starting'))

        adoutput_list = []
        for ad in rc.get_inputs(style='AD'):
            if ad.phu_get_key_value('STDSTRUC'):
                log.warning('%s has already been processed by standardizeStructure' %
                            (ad.filename))
                adoutput_list.append(ad)
                continue

            ad = sdz.standardize_structure_gmos(adinput=ad, add_mdf=rc['add_mdf'])
            adoutput_list.append(ad[0])

        rc.report_output(output)
        yield rc
    
    def subtractBias(self, rc):
        """
        This primitive will subtract the biases from the inputs using the 
        CL script gireduce.
        
        WARNING: The gireduce script used here replaces the previously 
        calculated DQ frames with its own versions.  This may be corrected 
        in the future by replacing the use of the gireduce
        with a Python routine to do the bias subtraction.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logType=rc['logType'],
                                  logLevel=rc['logLevel'])
        log.debug(gt.log_message('primitive', 'subtractBias', 'starting'))

        adoutput_list = []
        for ad in rc.get_inputs(style='AD'):
            if ad.phu_get_key_value('SUBBIAS'):
                log.warning('%s has already been processed by subtractBias' %
                            (ad.filename))
                adoutput_list.append(ad)
                continue

            #processedBias = AstroData(rc.get_cal(ad,'bias'))
            ####################BULL CRAP FOR TESTING ########################## 
            from copy import deepcopy
            processedBias = deepcopy(ad)
            processedBias.filename = 'TEMPNAMEforBIAS.fits'
            processedBias.phu_set_key_value('ORIGNAME','TEMPNAMEforBIAS.fits')
            processedBias.history_mark(key='GBIAS', 
                              comment='fake key to trick CL that GBIAS was ran')
            ####################################################################
            log.status('Using bias '+processedBias.filename+' to correct the inputs')
            ad = pp.subtract_biasNEW(adinput=ad, 
                                     bias=processedBias)
            adoutput_list.append(ad[0])

        rc.report_output(output)

        yield rc
    
    def validateData(self, rc):
        """
        This primitive is used to validate GMOS data, specifically. It will
        ensure the data is not corrupted or in an odd format that will affect
        later steps in the reduction process. If there are issues with the
        data, the flag 'repair' can be used to turn on the feature to repair it
        or not (e.g., validateData(repair=True)). 
        
        :param loglevel: Verbosity setting for log messages to the screen.
                         0 = nothing to screen, 6 = everything to screen. OR
                         the message level as a string (i.e., 'critical',
                         'status', 'fullinfo' ...)
        :type loglevel: integer or string
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc['logType'],
                                  logLevel=rc['logLevel'])
        # Log the standard 'starting primitive' debug message
        log.debug(gt.log_message('primitive', 'validateData', 'starting'))

        adoutput_list = []
        for ad in rc.get_inputs(style='AD'):
            if ad.phu_get_key_value('VALDATA'):
                log.warning('%s has already been processed by validateData' %
                            (ad.filename))
                adoutput_list.append(ad)
                continue

            ad = sdz.validate_data_gmos(adinput=ad, repair=rc['repair'])
            adoutput_list.append(ad[0])

        rc.report_output(output)
        yield rc
