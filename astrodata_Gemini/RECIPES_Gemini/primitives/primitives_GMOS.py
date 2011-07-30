import os, shutil, sys
from astrodata import Errors
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from astrodata.data import AstroData
from gempy import geminiTools as gt
from gempy.geminiCLParDicts import CLDefaultParamsDict
from gempy.science import preprocessing as pp
from gempy.science import resample as rs
from gempy.science import standardization as sdz
from primitives_GEMINI import GEMINIPrimitives

class GMOSPrimitives(GEMINIPrimitives):
    """
    This is the class containing all of the primitives for the GMOS level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'GEMINIPrimitives'.
    """
    astrotype = "GMOS"
    
    def init(self, rc):
        GEMINIPrimitives.init(self, rc)
        return rc

    def biasCorrect(self,rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "biasCorrect", "starting"))

        # Get processed biases for the input
        rc.run("getProcessedBias")

        # Loop over each input AstroData object in the input list to
        # test whether it's appropriate to try to subtract the bias
        sub_bias = True
        for ad in rc.get_inputs(style="AD"):

            # Check whether the subtractBias primitive has been run previously
            if ad.phu_get_key_value("SUBBIAS"):
                if rc["context"]=="QA":
                    sub_bias = False
                    log.warning("Files have already been processed by " +
                                "biasCorrect; no further bias " +
                                "correction performed")
                    rc.report_output(rc.get_inputs(style="AD"))
                    break
                else:
                    raise Errors.PrimitiveError("Files have already been " +
                                                "processed by " +
                                                "biasCorrect")

            # Test to see if we found a bias
            bias = AstroData(rc.get_cal(ad, "processed_bias"))
            if bias.filename is None:
                if rc['context']=="QA":
                    sub_bias = False
                    log.warning("No processed biases found; no bias " +
                                "correction performed")
                    rc.report_output(rc.get_inputs(style="AD"))
                    break
                    
                else:
                    raise Errors.PrimitiveError("No processed biases found")

        # If no errors found, subtract the bias frame
        if sub_bias:
            rc.run("subtractBias")

        yield rc

    def display(self, rc):
        """ 
        This is a primitive for displaying GMOS data. It utilizes the IRAF
        routine gdisplay and requires DS9 to be running before this primitive
        is called.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "display", "starting"))
        
        # Loading and bringing the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader()
        
        # Ensuring image buffer is large enough to handle GMOS images
        pyraf.iraf.set(stdimage="imtgmos")
        
        for i in range(0, len(rc.inputs)):
            # Retrieving the input object for this increment from the RC 
            inputRecord = rc.inputs[i]
            
            # Creating a dictionary of the parameters set by 
            # definition of the primitive 
            clPrimParams = {
                "image"         :inputRecord.filename,
                # Using the increment value (+1) for the frame value
                "frame"         :i+1,
                "fl_imexam"     :no,
                # Retrieving the observatory key from the PHU
                "observatory"   :inputRecord.ad.phu_get_key_value("OBSERVAT")
                }
            
            # Grabbing the default parameters dictionary and updating 
            # it with the above dictionary
            clParamsDict = CLDefaultParamsDict("gdisplay")
            clParamsDict.update(clPrimParams)
            
            # Logging the values in the prim parameter dictionaries
            gt.logDictParams(clPrimParams)
            
            log.debug("Calling the gdisplay CL script for input list %s" \
                      % (inputRecord.filename))
            
            try:
                # this version had the display id conversion code which we'll
                # need to redo code below just uses the loop index as frame
                # number
                #gemini.gmos.gdisplay(inputRecord.filename,
                #                     ds.displayID2frame(rq.dis_id),
                #                     fl_imexam=iraf.no,
                #                     Stdout = coi.get_iraf_stdout(),
                #                     Stderr = coi.get_iraf_stderr() )
                gemini.gmos.gdisplay(**clParamsDict)
                
                if gemini.gmos.gdisplay.status:
                    raise PrimitiveError("gdisplay failed for input %s" \
                                         % inputRecord.filename)
                else:
                    log.fullinfo("Exited the gdisplay CL script successfully")
            
            except:
                # This exception should allow for a smooth exiting if there is
                # an error with gdisplay, most likely due to DS9 not running
                # yet
                log.error("Unable to display %s" % (inputRecord.filename))
        
        yield rc
    
    def mosaicDetectors(self,rc):
        """
        This primitive will mosaic the SCI frames of the input images, along
        with the VAR and DQ frames if they exist.
        
        :param tile: tile images instead of mosaic
        :type tile: Python boolean (True/False), default is False
        
        :param interpolator: Type of interpolation function to use accross
                             the chip gaps. Options: 'linear', 'nearest',
                             'poly3', 'poly5', 'spine3', 'sinc'
        :type interpolator: string
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "mosaicDetectors", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            
            # Check whether the mosaicDetectors primitive has been run
            # previously
            if ad.phu_get_key_value("MOSAIC"):
                log.warning("%s has already been processed by " \
                            "mosaicDetectors" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # If the input AstroData object only has one extension, there is no
            # need to mosaic the detectors
            if ad.count_exts("SCI") == 1:
                log.warning("The input AstroData object only has one " \
                            "extension so there is no need to mosaic the " \
                            "detectors" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Call the mosaic_detectors user level function
            ad = rs.mosaic_detectors(adinput=ad, tile=rc["tile"],
                                     interpolator=rc["interpolator"])
            
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def overscanCorrect(self,rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "overscanCorrect", "starting"))

        # Loop over each input AstroData object in the input list to
        # test whether it's appropriate to try to subtract/trim the overscan
        sub_over = True
        for ad in rc.get_inputs(style="AD"):

            # Check whether the subtractOverscan or trimOverscan 
            # primitives have been run previously
            if (ad.phu_get_key_value("SUBOVER") or 
                ad.phu_get_key_value("TRIMOVER")):
                if rc["context"]=="QA":
                    sub_over = False
                    log.warning("Files have already been processed by " +
                                "overscanCorrect; no further overscan " +
                                "correction performed")
                    rc.report_output(rc.get_inputs(style="AD"))
                    break
                else:
                    raise Errors.PrimitiveError("Files have already been " +
                                                "processed by " +
                                                "overscanCorrect")

        # If no errors found, subtract the overscan and trim it
        if sub_over:
            recipe_list = ["subtractOverscan",
                           "trimOverscan"]
            rc.run("\n".join(recipe_list))

        yield rc

    def standardizeHeaders(self,rc):
        """
        This primitive is used to update and add keywords to the headers of the
        input dataset. First, it calls the standardize_headers_gemini user
        level function to update Gemini specific keywords and then updates GMOS
        specific keywords.
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
        
        :param attach_mdf: A flag to turn on/off appending the appropriate MDF 
                           file to the inputs.
        :type attach_mdf: Python boolean (True/False)
                          default: True
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
                                                attach_mdf=rc["attach_mdf"],
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
        The subtractBias primitive will subtract the science extension of the
        input bias frames from the science extension of the input science
        frames. The variance and data quality extension will be updated, if
        they exist.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "subtractBias", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            
            # Check whether the subtractBias primitive has been run previously
            if ad.phu_get_key_value("SUBBIAS"):
                log.warning("%s has already been processed by subtractBias" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Retrieve the appropriate bias
            bias = AstroData(rc.get_cal(ad, "processed_bias"))
            
            # If no appropriate bias is found, it is ok not to subtract the
            # bias 
            if not bias:
                log.warning("An appropriate bias for %s could not be found, " \
                            "so no bias will be subtracted" % (ad.filename))
                
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Call the subtract_bias user level function
            ad = pp.subtract_bias(adinput=ad, bias=bias)
            
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
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
                                '[2:25,1:2304],[2:25,1:2304],
                                [1032:1055,1:2304]'
                                is ideal for 2x2 GMOS data.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "subtractOverscan", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            
            # Check whether the subtractOverscan primitive has been run
            # previously
            if ad.phu_get_key_value("SUBOVER"):
                log.warning("%s has already been processed by " \
                            "subtractOverscan" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Call the subtract_overscan_gmos user level function
            ad = pp.subtract_overscan_gmos(
                adinput=ad, overscan_section=rc["overscan_section"])
            
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def trimOverscan(self,rc):
        """
        The trimOverscan primitive trims the overscan region from the input
        AstroData object and updates the headers.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "trimOverscan", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            
            # Check whether the trimOverscan primitive has been run previously
            if ad.phu_get_key_value("TRIMOVER"):
                log.warning("%s has already been processed by trimOverscan" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Call the trim_overscan user level function
            ad = pp.trim_overscan(adinput=ad)
            
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
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
            ad = sdz.validate_data_gmos(adinput=ad, repair=rc["repair"])
            
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
