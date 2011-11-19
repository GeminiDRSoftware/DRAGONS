import os
import numpy as np
from astrodata import AstroData
from astrodata import Errors
from astrodata import Lookups
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from gempy import geminiTools as gt
from gempy.science import preprocessing as pp
from gempy.science import resample as rs
from gempy.science import display as ds
from gempy.science import standardization as sdz
from gempy import managers as mgr
from gempy.geminiCLParDicts import CLDefaultParamsDict
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
    
    def display(self,rc):
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "display", "starting"))
        
        # Loop over each input AstroData object in the input list
        frame = rc["frame"]
        for ad in rc.get_inputs_as_astrodata():
            
            if frame>16:
                log.warning("Too many images; only the first 16 are displayed.")
                break

            threshold = rc["threshold"]
            if threshold is None:
                # Get the pre-defined threshold for the given detector type
                # and specific use case, i.e., display; using a look up
                # dictionary (table)
                gmosThresholds = Lookups.get_lookup_table(
                    "Gemini/GMOS/GMOSThresholdValues", "gmosThresholds")
                
                # Read the detector type from the phu
                detector_type = ad.phu_get_key_value("DETTYPE")

                # Form the key
                threshold_key = ("display", detector_type)
                if threshold_key in gmosThresholds:
                    # This is an integer with units ADU
                    threshold = gmosThresholds[threshold_key]
                else:
                    raise Errors.TableKeyError()
                
            try:
                ad = ds.display_gmos(adinput=ad,
                                     frame=frame,
                                     extname=rc["extname"],
                                     zscale=rc["zscale"],
                                     threshold=threshold)
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
        
        :param interpolate_gaps: Interpolate across gaps?
        :type interpolate_gaps: Python boolean (True/False)

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
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["mosaicDetectors"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Load the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader()

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the mosaicDetectors primitive has been run
            # previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by mosaicDetectors" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # If the input AstroData object only has one extension, there is no
            # need to mosaic the detectors
            if ad.count_exts("SCI") == 1:
                log.stdinfo("No changes will be made to %s, since it " \
                            "contains only one extension" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Get the necessary parameters from the RC
            tile = rc["tile"]
            interpolate_gaps = rc["interpolate_gaps"],
            interpolator = rc["interpolator"]
            
            # Get BUNIT, OVERSCAN,and AMPNAME from science extensions 
            # (gmosaic wipes out these keywords, they need to 
            # be restored after runnning it)
            bunit = None
            overscan = []
            ampname = []
            for ext in ad["SCI"]:
                ext_bunit = ext.get_key_value("BUNIT")
                if bunit is None:
                    bunit = ext_bunit
                else:
                    if ext_bunit!=bunit:
                        raise Errors.ScienceError("BUNIT needs to be the" +
                                                  "same for all extensions")
                ext_overscan = ext.get_key_value("OVERSCAN")
                if ext_overscan is not None:
                    overscan.append(ext_overscan)

                ext_ampname = ext.get_key_value("AMPNAME")
                if ext_ampname is not None:
                    ampname.append(ext_ampname)

            if len(overscan)>0:
                avg_overscan = np.mean(overscan)
            else:
                avg_overscan = None

            if len(ampname)>0:
                all_ampname = ",".join(ampname)
            else:
                all_ampname = None

            # Save detector section from 1st extension
            # FIXME - this assumes extensions are in order
            old_detsec = ad["SCI",1].detector_section().as_list()

            # Determine whether VAR/DQ needs to be propagated
            if (ad.count_exts("VAR") == 
                ad.count_exts("DQ") == 
                ad.count_exts("SCI")):
                fl_vardq=yes
            else:
                fl_vardq=no
            
            # Prepare input files, lists, parameters... for input to 
            # the CL script
            clm=mgr.CLManager(imageIns=ad, suffix="_out", 
                              funcName="mosaicDetectors", log=log)
            
            # Check the status of the CLManager object, 
            # True=continue, False= issue warning
            if not clm.status: 
                raise Errors.ScienceError("One of the inputs has not been " +
                                          "prepared, the " + 
                                          "mosaic_detectors function " +
                                          "can only work on prepared data.")
            
            # Parameters set by the mgr.CLManager or the 
            # definition of the prim 
            clPrimParams = {
                # Retrieve the inputs as a string of filenames
                "inimages"    :clm.imageInsFiles(type="string"),
                "outimages"   :clm.imageOutsFiles(type="string"),
                # Set the value of FL_vardq set above
                "fl_vardq"    :fl_vardq,
                # This returns a unique/temp log file for IRAF 
                "logfile"     :clm.templog.name,
                }
            # Parameters from the Parameter file adjustable by the user
            clSoftcodedParams = {
                # pyrafBoolean converts the python booleans to pyraf ones
                "fl_paste"    :mgr.pyrafBoolean(tile),
                "fl_fixpix"   :mgr.pyrafBoolean(interpolate_gaps),
                #"fl_clean"    :mgr.pyrafBoolean(False),
                "geointer"    :interpolator,
                }
            # Grab the default params dict and update it with 
            # the two above dicts
            clParamsDict = CLDefaultParamsDict("gmosaic")
            clParamsDict.update(clPrimParams)
            clParamsDict.update(clSoftcodedParams)
            
            # Log the parameters that were not defaults
            log.fullinfo("\nParameters set automatically:", 
                         category="parameters")
            # Loop through the parameters in the clPrimParams dictionary
            # and log them
            mgr.logDictParams(clPrimParams)
            
            log.fullinfo("\nParameters adjustable by the user:", 
                         category="parameters")
            # Loop through the parameters in the clSoftcodedParams 
            # dictionary and log them
            mgr.logDictParams(clSoftcodedParams)

            gemini.gmos.gmosaic(**clParamsDict)
            
            if gemini.gmos.gmosaic.status:
                raise Errors.ScienceError("gireduce failed for inputs "+
                             clm.imageInsFiles(type="string"))
            else:
                log.fullinfo("Exited the gmosaic CL script successfully")
            
            # Rename CL outputs and load them back into memory 
            # and clean up the intermediate temp files written to disk
            # refOuts and arrayOuts are None here
            imageOuts, refOuts, arrayOuts = clm.finishCL()
            
            ad_out = imageOuts[0]
            ad_out.filename = ad.filename
            
            # Verify gmosaic was actually run on the file
            # then log file names of successfully reduced files
            if ad_out.phu_get_key_value("GMOSAIC"): 
                log.fullinfo("File "+ad_out.filename+\
                            " was successfully mosaicked")

            # Get new DATASEC keyword, using the full shape of the
            # image extension
            data_shape = ad_out["SCI",1].data.shape
            new_datasec = "[1:%i,1:%i]" % (data_shape[1],
                                           data_shape[0])

            # Get new DETSEC keyword
            xbin = ad_out.detector_x_bin()
            if xbin is not None:
                unbin_width = data_shape[1] * xbin
            else:
                unbin_width = data_shape[1]
            if old_detsec is not None:
                new_detsec = "[%i:%i,%i:%i]" % (old_detsec[0]+1,
                                                old_detsec[0]+unbin_width,
                                                old_detsec[2]+1,old_detsec[3])
            else:
                new_detsec = ""

            # Get comment for new ampname
            if all_ampname is not None:
                # These ampnames can be long, so truncate
                # the comment by hand to avoid the error
                # message from pyfits
                ampcomment = self.keyword_comments["AMPNAME"]
                if len(all_ampname)>=65:
                    ampcomment = ""
                else:
                    ampcomment = ampcomment[0:65-len(all_ampname)]
            else:
                ampcomment = ""

            # Restore BUNIT, OVERSCAN, AMPNAME, DETSEC, DATASEC,CCDSEC
            # keywords to science extension header
            for ext in ad_out["SCI"]:
                if bunit is not None:
                    ext.set_key_value("BUNIT",bunit,
                                      comment=self.keyword_comments["BUNIT"])
                if avg_overscan is not None:
                    ext.set_key_value("OVERSCAN",avg_overscan,
                                      comment=self.keyword_comments["OVERSCAN"])

                if all_ampname is not None:
                    ext.set_key_value("AMPNAME",all_ampname,
                                      comment=ampcomment)

                ext.set_key_value("DETSEC",new_detsec,
                                  comment=self.keyword_comments["DETSEC"])

                ext.set_key_value("CCDSEC",new_detsec,
                                  comment=self.keyword_comments["CCDSEC"])

                ext.set_key_value("DATASEC",new_datasec,
                                  comment=self.keyword_comments["DATASEC"])

            # Restore BUNIT, DETSEC, DATASEC, CCDSEC, AMPNAME to VAR ext also
            if ad_out["VAR"] is not None:
                for ext in ad_out["VAR"]:
                    if bunit is not None:
                        ext.set_key_value("BUNIT","%s*%s" % (bunit,bunit),
                                        comment=self.keyword_comments["BUNIT"])
                    if all_ampname is not None:
                        ext.set_key_value("AMPNAME",all_ampname,
                                          comment=ampcomment)
                    ext.set_key_value("DETSEC",new_detsec,
                                      comment=self.keyword_comments["DETSEC"])
                    ext.set_key_value("CCDSEC",new_detsec,
                                      comment=self.keyword_comments["CCDSEC"])
                    ext.set_key_value("DATASEC",new_datasec,
                                      comment=self.keyword_comments["DATASEC"])


            # Change type of DQ plane back to int16
            # (gmosaic sets it to float32)
            # and restore DETSEC, DATASEC, CCDSEC
            if ad_out["DQ"] is not None:
                for ext in ad_out["DQ"]:
                    ext.data = ext.data.astype(np.int16)
                    ext.set_key_value("DETSEC",new_detsec,
                                      comment=self.keyword_comments["DETSEC"])
                    ext.set_key_value("CCDSEC",new_detsec,
                                      comment=self.keyword_comments["CCDSEC"])
                    ext.set_key_value("DATASEC",new_datasec,
                                      comment=self.keyword_comments["DATASEC"])

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad_out, keyword=timestamp_key)

            # Change the filename
            ad_out.filename = gt.fileNameUpdater(
                adIn=ad_out, suffix=rc["suffix"], strip=True)
            
            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad_out)
        
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
            timestamp_key = self.timestamp_keys["standardize_headers_gmos"]
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by standardizeHeaders" \
                            % (ad.filename))
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
            timestamp_key = self.timestamp_keys["standardize_structure_gmos"]
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by standardizeStructure" \
                            % (ad.filename))
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
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["subtractBias"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the subtractBias primitive has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by subtractBias" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Retrieve the appropriate bias
            bias = AstroData(rc.get_cal(ad, "processed_bias"))
            
            # If no appropriate bias is found, it is ok not to subtract the
            # bias in QA context; otherwise, raise error
            if bias.filename is None:
                if "qa" in rc.context:
                    log.warning("No changes will be made to %s, since no " \
                                "appropriate bias could be retrieved" \
                                % (ad.filename))
                
                    # Append the input AstroData object to the list of output
                    # AstroData objects without further processing
                    adoutput_list.append(ad)
                    continue
                else:
                    raise Errors.PrimitiveError("No processed bias found "\
                                                "for %s" % ad.filename)
            
            # Clip the bias frame to the size of the science data
            # For a GMOS example, this allows a full frame bias to
            # be used for a CCD2-only science frame. 
            bias = gt.clip_auxiliary_data(adinput=ad, aux=bias, 
                                          aux_type="cal")[0]

            # Check the inputs have matching binning and SCI shapes.
            gt.checkInputsMatch(adInsA=ad, adInsB=bias, 
                                check_filter=False) 

            log.fullinfo("Subtracting this bias from the input " \
                         "AstroData object (%s):\n%s" % (ad.filename, 
                                                         bias.filename))
            
            # Subtract the bias and handle VAR/DQ appropriately
            ad = ad.sub(bias)
            
            # Record the bias file used
            ad.phu_set_key_value("BIASIM", os.path.basename(bias.filename),
                                 comment=self.keyword_comments["BIASIM"])
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            
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
        
        Variance and DQ planes, if they exist, will be saved and restored
        after gireduce has been run.
        
        NOTE:
        The inputs to this function MUST be prepared.
        
        FOR FUTURE
        This function has many GMOS dependencies that would be great to work out
        so that this could be made a more general function (say at the Gemini
        level). In the future the parameters can be looked into and the CL 
        script can be upgraded to handle things like row based overscan
        calculations/fitting/modeling... vs the column based used right now, 
        add the model, nbiascontam,... params to the functions inputs so the 
        user can choose them for themselves.
        
        :param overscan_section: overscan_section parameter of format 
                   '[x1:x2,y1:y2],[x1:x2,y1:y2],[x1:x2,y1:y2]'
        :type overscan_section: string. 
                   eg: '[2:25,1:2304],[2:25,1:2304],[1032:1055,1:2304]' 
                   is ideal for 2x2 GMOS data. Default is None, which
                   causes default nbiascontam=4 columns to be used.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "subtractOverscan", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["subtractOverscan"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Load the pyraf related modules
        pyraf, gemini, yes, no = pyrafLoader() 

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the subtractOverscan primitive has been run
            # previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by subtractOverscan" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Get overscan_section parameter from the RC
            overscan_section = rc["overscan_section"]

            # Save VAR and DQ extensions
            var_ext = ad["VAR"]
            dq_ext = ad["DQ"]
            
            # Prepare input files, lists, parameters... for input to 
            # the CL script
            clm=mgr.CLManager(imageIns=ad, suffix=rc["suffix"],
                              funcName="subtractOverscan", log=log)
            
            # Check the status of the CLManager object, 
            # True=continue, False= issue warning
            if not clm.status:
                raise Errors.InputError("Inputs must be prepared")
            
            # Take care of the overscan_section->nbiascontam param
            if overscan_section is not None:
                nbiascontam = clm.nbiascontam(adinput, overscan_section)
                log.fullinfo("nbiascontam parameter was updated to = "+
                             str(nbiascontam))
            else: 
                # Do not try to calculate it, just use default value of 4.
                log.fullinfo("Using default nbiascontam parameter = 4")
                nbiascontam = 4
            
            # Parameters set by the mgr.CLManager or the definition 
            # of the primitive 
            clPrimParams = {
              "inimages"    :clm.imageInsFiles(type="string"),
              "gp_outpref"  :clm.prefix,
              "outimages"   :clm.imageOutsFiles(type="string"),
              # This returns a unique/temp log file for IRAF
              "logfile"     :clm.templog.name,
              "fl_over"     :yes, 
              "fl_trim"     :no,
              "outpref"     :"",
              "fl_vardq"    :no,
              "nbiascontam" :nbiascontam
                          }
            
            # Grab the default params dict and update it with 
            # the two above dicts
            clParamsDict = CLDefaultParamsDict("gireduce")
            clParamsDict.update(clPrimParams)
            
            # Log the parameters
            mgr.logDictParams(clParamsDict)
            
            log.debug("Calling the gireduce CL script for inputs "+
                      clm.imageInsFiles(type="string"))
            
            gemini.gmos.gireduce(**clParamsDict)
            
            if gemini.gmos.gireduce.status:
                raise Errors.ScienceError("gireduce failed for inputs "+
                             clm.imageInsFiles(type="string"))
            else:
                log.fullinfo("Exited the gireduce CL script successfully")
            
            # Rename CL outputs and load them back into memory, and 
            # clean up the intermediate tmp files written to disk
            # refOuts and arrayOuts are None here
            imageOuts, refOuts, arrayOuts = clm.finishCL() 
            ad = imageOuts[0]

            # Restore VAR/DQ planes; no additional propagation 
            # should be needed
            if dq_ext is not None:
                ad.append(dq_ext)
            if var_ext is not None:
                ad.append(var_ext)
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)

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
    
    def tileArrays(self,rc):
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "tileArrays", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Call the user level function,
            # which returns a list; take the first entry
            ad = rs.tile_arrays(adinput=ad,tile_all=rc["tile_all"])[0]
            
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
            timestamp_key = self.timestamp_keys["trim_overscan"]
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by trimOverscan" \
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
        or not (e.g., validateData(repair=True)). It currently just checks if
        there are 1, 2, 3, 4, 6, or 12 SCI extensions in the input.

        :param repair: Set to True (the default) to repair the data 
                       Note: this feature does not work yet.
        :type repair: Python boolean
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "validateData", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["validateData"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the validateData primitive has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by validateData" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Get the repair parameter from the RC
            repair = rc["repair"]

            # Validate the input AstroData object by ensuring that it has
            # 1, 2, 3, 4, 6 or 12 extensions
            valid_num_ext = [1, 2, 3, 4, 6, 12]
            num_ext = ad.count_exts("SCI")
            if num_ext not in valid_num_ext:
                if repair:
                    # This would be where we would attempt to repair the data 
                    raise Errors.Error("The 'repair' functionality is not " +
                                       "yet implemented")
                else:
                    raise Errors.Error("The number of extensions in %s do " +
                                       "match with the number of extensions " +
                                       "expected in raw GMOS data." \
                                           % (ad.filename))
                    
            else:
                log.fullinfo("The GMOS input file has been validated: %s " \
                             "contains %d extensions" % (ad.filename, num_ext))


            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            gt.mark_history(adinput=ad, keyword=self.timestamp_keys["prepare"])

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
