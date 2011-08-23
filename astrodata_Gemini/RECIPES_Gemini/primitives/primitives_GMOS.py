import os
import sys
import shutil
from astrodata import Errors
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from astrodata.data import AstroData
from gempy import geminiTools as gt
from gempy.geminiCLParDicts import CLDefaultParamsDict
from gempy.science import preprocessing as pp
from gempy.science import display as ds
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
        for ad in rc.get_inputs_as_astrodata():

            # Check whether the subtractBias primitive has been run previously
            if ad.phu_get_key_value("SUBBIAS"):
                if "QA" in rc.context:
                    sub_bias = False
                    log.warning("Files have already been processed by " +
                                "biasCorrect")
                    rc.report_output(rc.get_inputs_as_astrodata())
                    break
                else:
                    raise Errors.PrimitiveError("Files have already been " +
                                                "processed by " +
                                                "biasCorrect")

            # Test to see if we found a bias
            bias = AstroData(rc.get_cal(ad, "processed_bias"))
            if bias.filename is None:
                if "QA" in rc.context:
                    sub_bias = False
                    log.warning("No processed biases found")
                    rc.report_output(rc.get_inputs_as_astrodata())
                    break
                    
                else:
                    raise Errors.PrimitiveError("No processed biases found")

        # If no errors found, subtract the bias frame
        if sub_bias:
            rc.run("subtractBias")

        yield rc

    def display(self,rc):

        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "display", "starting"))

        
        # Loop over each input AstroData object in the input list
        frame = rc["frame"]
        for ad in rc.get_inputs_as_astrodata():

            try:
                ad = ds.display_gmos(adinput=ad,
                                     frame=frame,
                                     saturation=rc['saturation'])
            except:
                log.warning("Could not display %s" % ad.filename)

            frame+=1
        
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
        for ad in rc.get_inputs_as_astrodata():
            
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
                log.warning("Only one extension in %s" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Call the mosaic_detectors user level function,
            # which returns a list; take the first entry
            ad = rs.mosaic_detectors(adinput=ad, tile=rc["tile"],
                                     interpolator=rc["interpolator"])[0]
            
            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                             strip=True)

            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)
        
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
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the standardizeHeaders primitive has been run
            # previously
            if ad.phu_get_key_value("SDZHDRSI"):
                log.warning("%s has already been processed by " \
                            "standardizeHeaders" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Call the standardize_headers_gmos user level function,
            # which returns a list; take the first entry
            ad = sdz.standardize_headers_gmos(adinput=ad)[0]
            
            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                             strip=True)

            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)
        
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
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the standardizeStructure primitive has been run
            # previously
            if ad.phu_get_key_value("SDZSTRUC"):
                log.warning("%s has already been processed by " \
                            "standardizeStructure" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Call the standardize_structure_gmos user level function,
            # which returns a list; take the first entry
            ad = sdz.standardize_structure_gmos(adinput=ad,
                                                attach_mdf=rc["attach_mdf"],
                                                mdf=rc["mdf"])[0]
            
            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                             strip=True)

            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)

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
        for ad in rc.get_inputs_as_astrodata():
            
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
            if bias.filename is None:
                log.warning("Could not find an appropriate bias for %s" % 
                            ad.filename)

                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Call the subtract_bias user level function,
            # which returns a list; take the first entry
            ad = pp.subtract_bias(adinput=ad, bias=bias)[0]
            
            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                             strip=True)

            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)
        
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
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the subtractOverscan primitive has been run
            # previously
            if ad.phu_get_key_value("SUBOVER"):
                log.warning("%s has already been processed by " \
                            "subtractOverscan" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Call the subtract_overscan_gmos user level function,
            # which returns a list; take the first entry
            ad = pp.subtract_overscan_gmos(adinput=ad, 
                                    overscan_section=rc["overscan_section"])[0]
            
            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                             strip=True)

            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)
        
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
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the trimOverscan primitive has been run previously
            if ad.phu_get_key_value("TRIMOVER"):
                log.warning("%s has already been processed by trimOverscan" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Call the trim_overscan user level function,
            # which returns a list; take the first entry
            ad = pp.trim_overscan(adinput=ad)[0]
            
            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                             strip=True)

            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)
        
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
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the validateData primitive has been run previously
            if ad.phu_get_key_value("VALDATA"):
                log.warning("%s has already been processed by validateData" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Call the validate_data_gmos user level function,
            # which returns a list; take the first entry
            ad = sdz.validate_data_gmos(adinput=ad, repair=rc["repair"])[0]
            
            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                             strip=True)

            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
