# Author: Kyle Mede. 2010
# Skeleton originally written by Craig Allen, callen@gemini.edu

import os, shutil, sys
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from astrodata.ConfigSpace import lookup_path
from astrodata.data import AstroData
from astrodata import Errors
from gempy import geminiTools as gt
from gempy.science import preprocessing as pp
from gempy.science import resample as rs
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
     
    def display(self, rc):
        """ 
        This is a primitive for displaying GMOS data.
        It utilizes the IRAF routine gdisplay and requires DS9 to be running
        before this primitive is called.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
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
                    log.fullinfo('Exited the gdisplay CL script successfully')
                        
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
        
        :param interpolation: Type of interpolation function to use accross
                             the chip gaps
        :type interpolation: string, options: 'linear', 'nearest', 'poly3', 
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

            # Check for either a 3-extension (E2V) or 
            # 12-extension (Hamamatsu) file
            nsciext = ad.count_exts('SCI')
            if nsciext!=3 and nsciext!=12:
                log.warning('%s cannot be mosaicked.' % ad.filename)
                adoutput_list.append(ad)
                continue

            ad = rs.mosaic_detectors(adinput=ad, 
                                     tile=rc['tile'], 
                                     interpolation=rc['interpolation'])            
            adoutput_list.append(ad[0])
                
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)                
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
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "standardizeHeaders",
                                 "starting"))
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Check whether the standardizeHeaders primitive has been run
            # previously
            if ad.phu_get_key_value("SDZHDRSI"):
                log.warning("%s has already been processed by " \
                            "standardizeHeaders" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            # Call the standardize_headers_gmos user level function
            ad = sdz.standardize_headers_gmos(adinput=ad)
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        yield rc

    def standardizeStructure(self,rc):
        """
        This primitive will add an MDF to the
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
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "standardizeStructure",
                                 "starting"))
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Check whether the standardizeStructure primitive has been run
            # previously
            if ad.phu_get_key_value("SDZSTRUC"):
                log.warning("%s has already been processed by " \
                            "standardizeStructure" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            # Call the standardize_structure_gmos user level function
            ad = sdz.standardize_structure_gmos(adinput=ad,
                                                add_mdf=rc['add_mdf'],
                                                mdf=rc["mdf"])
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
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

            # Retrieve the appropriate bias
            bias = AstroData(rc.get_cal(ad,'bias'))

            # Take care of the case where there was no, or an invalid bias
            if bias is None or bias.count_exts("SCI") == 0:
                log.warning("Could not find an appropriate bias for %s" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue

            ad = pp.subtract_bias(adinput=ad, bias=bias)
            adoutput_list.append(ad[0])

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)

        yield rc
    
    def subtractOverscan(self,rc):
        """
        This primitive uses the CL script gireduce to subtract the overscan 
        from the input images.
        
        :param overscan_section: biassec parameter of format 
                                 '[#:#,#:#],[#:#,#:#],[#:#,#:#]'
        :type overscan_section: string. default: 
                                '[2:25,1:2304],[2:25,1:2304],[1032:1055,1:2304]'
                                is ideal for 2x2 GMOS data.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logType=rc['logType'],
                                  logLevel=rc['logLevel'])
        log.debug(gt.log_message('primitive', 'subtractOverscan', 'starting'))
        
        adoutput_list = []
        for ad in rc.get_inputs(style='AD'):
            if ad.phu_get_key_value('SUBOVER'):
                log.warning('%s has already been processed by subtractOverscan'%
                            (ad.filename))
                adoutput_list.append(ad)
                continue

            ad = pp.subtract_overscan_gmos(adinput=ad,
                                       overscan_section=rc['overscan_section'])
            adoutput_list.append(ad[0])

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)

        yield rc    

    def trimOverscan(self,rc):
        """
        This primitive uses AstroData to trim the overscan region 
        from the input images and update their headers.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 
                        'critical', 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logType=rc['logType'],
                                  logLevel=rc['logLevel'])

        log.debug(gt.log_message('primitive', 'trimOverscan', 'starting'))
        
        adoutput_list = []
        for ad in rc.get_inputs(style='AD'):
            if ad.phu_get_key_value('TRIMOVER'):
                log.warning('%s has already been processed by trimOverscan' %
                            (ad.filename))
                adoutput_list.append(ad)
                continue
            
            ad = pp.trim_overscan(adinput=ad)
            adoutput_list.append(ad[0])

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)   

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
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "validateData", "starting"))
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Check whether the validateData primitive has been run previously
            if ad.phu_get_key_value("VALDATA"):
                log.warning("%s has already been processed by validateData" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            # Call the validate_data_gmos user level function
            ad = sdz.validate_data_gmos(adinput=ad, repair=rc['repair'])
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
