import os
import datetime
import numpy as np
import pyfits as pf
import pywcs
from astrodata import AstroData
from astrodata import Errors
from astrodata import Lookups
from astrodata.adutils import logutils
from astrodata.adutils.gemutil import pyrafLoader
from gempy import gemini_tools as gt
from gempy import managers as mgr
from gempy.geminiCLParDicts import CLDefaultParamsDict
from primitives_GEMINI import GEMINIPrimitives
from gempy.eti.gireduceeti import GireduceETI
from gempy.eti.gmosaiceti import GmosaicETI
import time

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
    
    def mosaicDetectors(self,rc):
        """
        This primitive will mosaic the SCI frames of the input images, along
        with the VAR and DQ frames if they exist. It uses the the ETI and pyraf
        to call gmosaic from the gemini IRAF package.
        
        :param tile: tile images instead of mosaic
        :type tile: Python boolean (True/False), default is False
        
        :param interpolate_gaps: Interpolate across gaps?
        :type interpolate_gaps: Python boolean (True/False)

        :param interpolator: Type of interpolation function to use accross
                             the chip gaps. Options: 'linear', 'nearest',
                             'poly3', 'poly5', 'spine3', 'sinc'
        :type interpolator: string
        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", "mosaicDetectors", "starting"))
        timestamp_key = self.timestamp_keys["mosaicDetectors"]
        adoutput_list = []
        
        for ad in rc.get_inputs_as_astrodata():
            
            # Validate Data
            if (ad.phu_get_key_value('GPREPARE')==None) and \
                (ad.phu_get_key_value('PREPARE')==None):
                raise Errors.InputError("%s must be prepared" % ad.filename)

            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by mosaicDetectors" \
                            % (ad.filename))
                adoutput_list.append(ad)
                continue
            if ad.count_exts("SCI") == 1:
                log.stdinfo("No changes will be made to %s, since it " \
                            "contains only one extension" % (ad.filename))
                adoutput_list.append(ad)
                continue
            
            # Save keywords for restoration after gmosaic
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

            if len(ampname)>0:
                all_ampname = ",".join(ampname)
            else:
                all_ampname = None

            if len(overscan)>0:
                avg_overscan = np.mean(overscan)
            else:
                avg_overscan = None

            # FIXME - this assumes extensions are in order
            old_detsec = ad["SCI",1].detector_section().as_list()

            # Instantiate ETI and then run the task
            gmosaic_task = GmosaicETI(rc,ad)
            ad_out = gmosaic_task.run()

            # Get new DATASEC keyword, using the full shape
            data_shape = ad_out["SCI",1].data.shape
            new_datasec = "[1:%i,1:%i]" % (data_shape[1], data_shape[0])

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

            # To avoid pyfits error truncate long comments
            if all_ampname is not None:
                ampcomment = self.keyword_comments["AMPNAME"]
                if len(all_ampname)>=65:
                    ampcomment = ""
                else:
                    ampcomment = ampcomment[0:65-len(all_ampname)]
            else:
                ampcomment = ""

            # Restore keywords to science extension header
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

            # Change type of DQ plane back to int16 (gmosaic sets float32)
            # , restore DETSEC, DATASEC, CCDSEC, and replace any -1 values with 1 
            # (gmosaic marks chip gaps with -1 if fixpix=no and clean=no)
            if ad_out["DQ"] is not None:
                for ext in ad_out["DQ"]:
                    ext.data = ext.data.astype(np.int16)
                    ext.data = np.where(ext.data<0,1,ext.data)
                    ext.set_key_value("DETSEC",new_detsec,
                                      comment=self.keyword_comments["DETSEC"])
                    ext.set_key_value("CCDSEC",new_detsec,
                                      comment=self.keyword_comments["CCDSEC"])
                    ext.set_key_value("DATASEC",new_datasec,
                                      comment=self.keyword_comments["DATASEC"])

            gt.mark_history(adinput=ad_out, keyword=timestamp_key)
            adoutput_list.append(ad_out)
        rc.report_output(adoutput_list)
        yield rc
    
    def mosaicDetectorsDEPRECATED(self,rc):
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
        log = logutils.get_logger(__name__)
        
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
            interpolate_gaps = rc["interpolate_gaps"]
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

            if len(ampname)>0:
                all_ampname = ",".join(ampname)
            else:
                all_ampname = None

            if len(overscan)>0:
                avg_overscan = np.mean(overscan)
            else:
                avg_overscan = None

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
            
            # Check whether the default statistics section can be used,
            # if not, turn off fl_clean to avoid crashes
            # Current default is [2150:3970,100:4400], unbinned
            xbin = ad.detector_x_bin().as_pytype()
            ybin = ad.detector_x_bin().as_pytype()
            default_sec = [2150/xbin,3970/xbin,100/ybin,4400/ybin]
            fl_clean = yes
            for ext in ad["SCI"]:
                shape = ext.data.shape
                
                # Check y only, for 6-amp data the x will always
                # be too small.
                if shape[0]<default_sec[3]:
                    fl_clean = no
                    break

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
                # Set the clean parameter
                "fl_clean"    :fl_clean,
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
            log.fullinfo("\nParameters set automatically:")
            # Loop through the parameters in the clPrimParams dictionary
            # and log them
            mgr.logDictParams(clPrimParams)
            
            log.fullinfo("\nParameters adjustable by the user:")
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
            # and replace any -1 values with 1 
            # (gmosaic marks chip gaps with -1 if fixpix=no and clean=no)
            if ad_out["DQ"] is not None:
                for ext in ad_out["DQ"]:
                    ext.data = ext.data.astype(np.int16)
                    ext.data = np.where(ext.data<0,1,ext.data)
                    ext.set_key_value("DETSEC",new_detsec,
                                      comment=self.keyword_comments["DETSEC"])
                    ext.set_key_value("CCDSEC",new_detsec,
                                      comment=self.keyword_comments["CCDSEC"])
                    ext.set_key_value("DATASEC",new_datasec,
                                      comment=self.keyword_comments["DATASEC"])

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad_out, keyword=timestamp_key)

            # Change the filename
            ad_out.filename = gt.filename_updater(
                adinput=ad_out, suffix=rc["suffix"], strip=True)
            
            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad_out)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def standardizeInstrumentHeaders(self,rc):
        """
        This primitive is used to update and add keywords specific
        to GMOS data.
        """
        
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "standardizeInstrumentHeaders",
                                 "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["standardizeInstrumentHeaders"]

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Look up static bias levels
        gmosampsBias, gmosampsBiasBefore20060831 = \
            Lookups.get_lookup_table("Gemini/GMOS/GMOSAmpTables",
                                     "gmosampsBias",
                                     "gmosampsBiasBefore20060831")

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the standardizeInstrumentHeaders primitive
            # has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by "\
                            "standardizeInstrumentHeaders" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Update the keywords in the headers that are specific to GMOS
            log.fullinfo("Updating keywords that are specific to GMOS")

            # Pixel scale
            gt.update_key_from_descriptor(
                adinput=ad, descriptor="pixel_scale()", extname="SCI")

            # Read noise
            gt.update_key_from_descriptor(
                adinput=ad, descriptor="read_noise()", extname="SCI")

            # Gain setting
            gt.update_key_from_descriptor(
                adinput=ad, descriptor="gain_setting()", extname="SCI")

            # Gain
            gt.update_key_from_descriptor(
                adinput=ad, descriptor="gain()", extname="SCI")
            
            # Bias level
            if "qa" in rc.context:
                # Get the bias level from static tables
                gt.update_key_from_descriptor(
                    adinput=ad, descriptor="bias_level()", extname="SCI")
            else:
                # For science quality, get the bias from a median
                # of the overscan region.  Assume that data has not
                # yet been trimmed or processed to remove bias level,
                # and that units are ADU
                oversec_dv = ad.overscan_section()
                if oversec_dv is None:
                    # Use the static bias levels
                    gt.update_key_from_descriptor(
                        adinput=ad, descriptor="bias_level()", extname="SCI")
                else:
                    oversec_dict = oversec_dv.dict_val

                    detector_type = ad.phu_get_key_value("DETTYPE")
        
                    # The type of CCD determines the number of contaminated
                    # columns in the overscan region
                    if detector_type=="SDSU II CCD":
                        nbiascontam = 4
                    elif detector_type=="SDSU II e2v DD CCD42-90":
                        nbiascontam = 5
                    elif detector_type=="S10892-01":
                        nbiascontam = 4
                    else:
                        nbiascontam = 4

                    for ext in ad["SCI"]:
                        dict_key = (ext.extname(),ext.extver())
                        oversec = oversec_dict[dict_key]

                        # Don't include columns at edges
                        if oversec[0]==0:
                            # Overscan region is on the left
                            oversec[1]-=nbiascontam
                            oversec[0]+=1
                        else:
                            # Overscan region is on the right
                            oversec[0]+=nbiascontam
                            oversec[1]-=1
                    
                        # Extract overscan data.  In numpy arrays, 
                        # y indices come first.
                        overdata = ext.data[oversec[2]:oversec[3],
                                            oversec[0]:oversec[1]]
                
                        bias_level = np.median(overdata)
                        ext.set_key_value(
                            "RAWBIAS",bias_level,
                            comment=self.keyword_comments["RAWBIAS"])

            # Saturation level
            gt.update_key_from_descriptor(
                adinput=ad, descriptor="saturation_level()", extname="SCI")

            # Dispersion axis
            if "IMAGE" not in ad.types:
                gt.update_key_from_descriptor(
                    adinput=ad, descriptor="dispersion_axis()", extname="SCI")

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"], 
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
        This function ensures the MEF structure of GMOS data is ready for
        further processing, through adding an MDF if necessary. 
        Appropriately all SPECT type data should have an MDF added, while
        that of IMAGE should not. If input contains mixed types of GMOS data
        (ie. some IMAGE and some SPECT), then only those of type SPECT will
        have MDFs attached.

        :param attach_mdf: A flag to turn on/off appending the appropriate MDF 
                           file to the inputs.
        :type attach_mdf: Python boolean (True/False)
                          default: True
                  
        :param mdf: A file name (with path) of the MDF file to append onto the
                     input(s).
                     Note: If there are multiple inputs and one mdf
                     provided, then the same MDF will be applied to all inputs;
                     else the mdf must be in a list of match the length of
                     the inputs and the inputs must ALL be of type SPECT.
        :type mdf: String, or list of strings
        """
        
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "standardizeStructure",
                                 "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["standardizeStructure"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # First run addMDF if necessary to attach mdf files to the input
        for ad in rc.get_inputs_as_astrodata():
            if rc["attach_mdf"]:
                # Get the mdf parameter from the RC
                mdf = rc["mdf"]
                if mdf is not None:
                    rc.run("addMDF(mdf=%s)" % mdf)
                else:
                    rc.run("addMDF")

                # It only needs to run once, on all input, so break loop
                break

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            # Check whether the standardizeStructure primitive has been run
            # previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by standardizeStructure" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)
                
            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"], 
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
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "subtractBias", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["subtractBias"]

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Check for a user-supplied bias
        adinput = rc.get_inputs_as_astrodata()
        bias_param = rc["bias"]
        bias_dict = None
        if bias_param is not None:
            # The user supplied an input to the bias parameter
            if not isinstance(bias_param, list):
                bias_list = [bias_param]
            else:
                bias_list = bias_param

            # Convert filenames to AD instances if necessary
            tmp_list = []
            for bias in bias_list:
                if type(bias) is not AstroData:
                    bias = AstroData(bias)
                tmp_list.append(bias)
            bias_list = tmp_list
            
            bias_dict = gt.make_dict(key_list=adinput, value_list=bias_list)
        
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            
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
            if bias_dict is not None:
                bias = bias_dict[ad]
            else:
                bias = rc.get_cal(ad, "processed_bias")

                # If no appropriate bias is found, it is ok not to subtract the
                # bias in QA context; otherwise, raise error
                if bias is None:
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
                else:
                    bias = AstroData(bias)

            # Check the inputs have matching binning and SCI shapes.
            try:
                gt.check_inputs_match(ad1=ad, ad2=bias, 
                                      check_filter=False) 
            except Errors.ToolboxError:
                # If not, try to clip the bias frame to the size of
                # the science data
                # For a GMOS example, this allows a full frame bias to
                # be used for a CCD2-only science frame. 
                bias = gt.clip_auxiliary_data(adinput=ad, aux=bias, 
                                              aux_type="cal")[0]

                # Check again, but allow it to fail if they still don't match
                gt.check_inputs_match(ad1=ad, ad2=bias, 
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
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"], 
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
        This primitive uses External Task Interface to gireduce to subtract 
        the overscan from the input images.
        
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
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", "subtractOverscan", "starting"))
        adinput = rc.get_inputs_as_astrodata()
        adoutput_list = []
        timestamp_key = self.timestamp_keys["subtractOverscan"]
        
        for ad in adinput:
            
            # Validate Data
            if (ad.phu_get_key_value('GPREPARE')==None) and \
                (ad.phu_get_key_value('PREPARE')==None):
                raise Errors.InputError("%s must be prepared" % ad.filename)
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by subtractOverscan" \
                            % (ad.filename))
                adoutput_list.append(ad)
                continue

            var_ext = ad["VAR"]
            dq_ext = ad["DQ"]
            objcat = ad["OBJCAT"]
            refcat = ad["REFCAT"]
            objmask = ad["OBJMASK"]
            
            # Instantiate ETI and then run the task
            gireduce_task = GireduceETI(rc,ad)
            adout = gireduce_task.run()
            
            if dq_ext is not None:
                adout.append(dq_ext)
            if var_ext is not None:
                adout.append(var_ext)
            if objcat is not None:
                adout.append(objcat)
            if refcat is not None:
                adout.append(refcat)
            if objmask is not None:
                adout.append(objmask)
            
            gt.mark_history(adinput=adout, keyword=timestamp_key)
            adoutput_list.append(adout)
        
        rc.report_output(adoutput_list)
        yield rc
   
    def subtractOverscanDEPRECATED(self,rc):
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
        log = logutils.get_logger(__name__)
        
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

            # Save VAR, DQ, OBJCAT, REFCAT, OBJMASK extensions
            var_ext = ad["VAR"]
            dq_ext = ad["DQ"]
            objcat = ad["OBJCAT"]
            refcat = ad["REFCAT"]
            objmask = ad["OBJMASK"]
            
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
                # Do not try to calculate it, just use default value
                # For e2vDD detectors, this is 5. Otherwise, use 4.
                detector_type = ad.phu_get_key_value("DETTYPE")
                if detector_type=="SDSU II e2v DD CCD42-90":
                    log.fullinfo("Using default nbiascontam parameter = 5")
                    nbiascontam = 5
                else:
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
            
            #from pprint import pprint
            #pprint(clParamsDict)
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

            # Restore non-SCI planes; no additional propagation 
            # should be needed
            if dq_ext is not None:
                ad.append(dq_ext)
            if var_ext is not None:
                ad.append(var_ext)
            if objcat is not None:
                ad.append(objcat)
            if refcat is not None:
                ad.append(refcat)
            if objmask is not None:
                ad.append(objmask)
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"], 
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
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "tileArrays", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["tileArrays"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            # Get the necessary parameters from the RC
            tile_all = rc["tile_all"]

            # Store PHU to pass to output AD
            # The original name must be stored first so that the
            # output AD can reconstruct it later
            ad.store_original_name()
            phu = ad.phu

            # Do nothing if there is only one science extension
            nsciext = ad.count_exts("SCI")
            if nsciext==1:
                log.fullinfo("Only one science extension found; " +
                             "no tiling done for %s" % ad.filename)
                adoutput_list.append(ad)
                continue

            # Flag to track whether input has changed
            changed = False

            # First trim off any overscan regions still present
            # so they won't get tiled with science data
            log.fullinfo("Trimming data to data section:")
            old_shape = " ".join(
                ["%i,%i" % ext.data.shape for ext in ad["SCI"]])
            ad = gt.trim_to_data_section(adinput=ad)[0]
            new_shape = " ".join(
                ["%i,%i" % ext.data.shape for ext in ad["SCI"]])
            if old_shape!=new_shape:
                changed = True

            # Make chip gaps to tile with science extensions if tiling all
            # Gap width comes from a lookup table
            gap_height = int(ad["SCI",1].data.shape[0])
            gap_width = _obtain_arraygap(adinput=ad)
            chip_gap = np.zeros((gap_height,gap_width))

            # Get the correct order of the extensions by sorting on
            # the first element in detector section
            # (raw ordering is whichever amps read out first)
            detsecs = ad.detector_section().as_list()
            if not isinstance(detsecs[0],list):
                detsecs = [detsecs]
            detx1 = [sec[0] for sec in detsecs]
            ampsorder = range(1,nsciext+1)
            orderarray = np.array(
                zip(ampsorder,detx1),dtype=[('ext',np.int),('detx1',np.int)])
            orderarray.sort(order='detx1')
            if np.all(ampsorder==orderarray['ext']):
                in_order = True
            else:
                ampsorder = orderarray['ext']
                in_order = False
                
            # Get array sections for determining when
            # a new array is found
            ccdsecs = ad.array_section().as_list()
            if not isinstance(ccdsecs[0],list):
                ccdsecs = [ccdsecs]
            if len(ccdsecs)!=nsciext:
                ccdsecs*=nsciext
            ccdx1 = [sec[0] for sec in ccdsecs]


            # Now, get the number of extensions per ccd

            # Initialize everything 
            ccd_data = {}
            amps_per_ccd = {}
            sci_data_list = []
            var_data_list = []
            dq_data_list = []
            mask_data_list = []
            num_ccd = 0
            ext_count = 1
            ampname = {}
            amplist = []
            refsec = {}
            mapping_dict = {}
            
            # Initialize these so that first extension will always
            # start a new CCD
            last_detx1 = detx1[ampsorder[0]-1]-1
            last_ccdx1 = ccdx1[ampsorder[0]-1]

            for i in ampsorder:
                sciext = ad["SCI",i]
                varext = ad["VAR",i]
                dqext = ad["DQ",i]
                maskext = ad["OBJMASK",i]

                this_detx1 = detx1[i-1]
                this_ccdx1 = ccdx1[i-1]

                amp = sciext.get_key_value("AMPNAME")

                if (this_detx1>last_detx1 and this_ccdx1<=last_ccdx1):
                    # New CCD found

                    # If not first extension, store current data lists
                    # (or, if tiling all CCDs together, add a chip gap)
                    if num_ccd>0:
                        if tile_all:
                            sci_data_list.append(
                                chip_gap.astype(np.float32))
                            if varext is not None:
                                var_data_list.append(
                                    chip_gap.astype(np.float32))
                            if dqext is not None:
                                # For the DQ plane, set the gap value
                                # to 1 (bad pixel)
                                dq_data_list.append(
                                    chip_gap.astype(np.int16)+1)
                            if maskext is not None:
                                mask_data_list.append(
                                    chip_gap.astype(np.int16))
                        else:
                            ccd_data[num_ccd] = {"SCI":sci_data_list,
                                                 "VAR":var_data_list,
                                                 "DQ":dq_data_list,
                                                 "OBJMASK":mask_data_list}
                            ampname[num_ccd] = amplist

                    # Increment CCD number and restart amps per ccd
                    num_ccd += 1
                    amps_per_ccd[num_ccd] = 1

                    # Start new data lists (or append if tiling all)
                    if tile_all:                            
                        sci_data_list.append(sciext.data)
                        if varext is not None:
                            var_data_list.append(varext.data)
                        if dqext is not None:
                            dq_data_list.append(dqext.data)
                        if maskext is not None:
                            mask_data_list.append(maskext.data)

                        # Keep the name of the amplifier
                        # (for later header updates)
                        amplist.append(amp)

                        # Keep ccdsec and detsec from first extension only
                        if num_ccd==1:
                            refsec[1] = {"CCD":ccdsecs[i-1],
                                         "DET":detsecs[i-1]} 
                    else:
                        sci_data_list = [sciext.data]
                        if varext is not None:
                            var_data_list = [varext.data]
                        if dqext is not None:
                            dq_data_list = [dqext.data]
                        if maskext is not None:
                            mask_data_list = [maskext.data]
                        amplist = [amp]
                        # Keep ccdsec and detsec from first extension
                        # of each CCD
                        refsec[num_ccd] = {"CCD":ccdsecs[i-1],
                                           "DET":detsecs[i-1]}
                else:
                    # Increment amps and append data
                    amps_per_ccd[num_ccd] += 1
                    amplist.append(amp)
                    sci_data_list.append(sciext.data)
                    if varext:
                        var_data_list.append(varext.data)
                    if dqext:
                        dq_data_list.append(dqext.data)
                    if maskext:
                        mask_data_list.append(maskext.data)
                    

                # If last iteration, store the current data lists
                if tile_all:
                    key = 1
                else:
                    key = num_ccd
                if ext_count==nsciext:
                    ccd_data[key] = {"SCI":sci_data_list,
                                     "VAR":var_data_list,
                                     "DQ":dq_data_list,
                                     "OBJMASK":mask_data_list}
                    ampname[key] = amplist

                # Keep track of which extensions ended up in
                # which CCD
                try:
                    mapping_dict[key].append(i)
                except KeyError:
                    mapping_dict[key] = [i]

                last_ccdx1 = this_ccdx1
                last_detx1 = this_detx1
                ext_count += 1

            if nsciext==num_ccd and in_order and not tile_all:
                # No reordering or tiling necessary, return input AD
                log.fullinfo("Only one amplifier per array; " +
                             "no tiling done for %s" % ad.filename)

                # If file has not changed, go to next file.  If it
                # has changed, set time stamps and change filename
                # at the end of the for-loop
                if changed:
                    adoutput = ad
                else:
                    adoutput_list.append(ad)
                    continue
            else:
                if not in_order:
                    log.fullinfo("Reordering data by detector section")
                if tile_all:
                    log.fullinfo("Tiling all data into one extension")
                elif nsciext!=num_ccd:
                    log.fullinfo("Tiling data into one extension per array")

                # Get header from the center-left extension of each CCD
                # (or the center-left of CCD2 if tiling all)
                # This is in order to get the most accurate WCS on CCD2
                ref_header = {}
                startextn = 1
                ref_shift = {}
                ref_shift_temp = 0
                total_shift = 0
                on_ext=0
                for ccd in range(1,num_ccd+1):
                    if tile_all:
                        key = 1
                        if ccd!=2:
                            startextn += amps_per_ccd[ccd]
                            continue
                    else:
                        key = ccd
                        total_shift = 0

                    refextn = ampsorder[int((amps_per_ccd[ccd]+1)/2.0-1)
                                        + startextn - 1]

                    # Get size of reference shift from 0,0 to
                    # start of reference extension
                    for data in ccd_data[key]["SCI"]:
                        # if it's a chip gap, add width to total, continue
                        if data.shape[1]==gap_width:
                            total_shift += gap_width
                        else:
                            on_ext+=1
                            # keep total up to now if it's the reference ext
                            if ampsorder[on_ext-1]==refextn:
                                ref_shift_temp = total_shift
                            # add in width of this extension
                            total_shift += data.shape[1]

                    # Get header from reference extension
                    dict = {}
                    for extname in ["SCI","VAR","DQ","OBJMASK"]:
                        ext = ad[extname,refextn]
                        if ext is not None:
                            header = ext.header
                        else:
                            header = None
                        dict[extname] = header
                    
                    if dict["SCI"] is None:
                        raise Errors.ScienceError("Header not found " +
                                                  "for reference " + 
                                                  "extension " +
                                                  "[SCI,%i]" % refextn)

                    ref_header[key] = dict
                    ref_shift[key] = ref_shift_temp

                    startextn += amps_per_ccd[ccd]

                # Make a new AD
                adoutput = AstroData()
                adoutput.filename = ad.filename
                adoutput.phu = phu

                # Stack data from each array together and
                # append to output AD
                if tile_all:
                    num_ccd = 1
                nextend = 0
                for ccd in range(1,num_ccd+1):
                    for extname in ["SCI","DQ","VAR","OBJMASK"]:
                        if (extname in ccd_data[ccd] and 
                            len(ccd_data[ccd][extname])>0):
                            data = np.hstack(ccd_data[ccd][extname])
                            header = ref_header[ccd][extname]
                            new_ext = AstroData(data=data,header=header)
                            new_ext.rename_ext(name=extname,ver=ccd)
                            adoutput.append(new_ext)
                        
                            nextend += 1

                # Update header keywords with appropriate values
                # for the new data set
                adoutput.phu_set_key_value(
                    "NSCIEXT",num_ccd,comment=self.keyword_comments["NSCIEXT"])
                adoutput.phu_set_key_value(
                    "NEXTEND",nextend,comment=self.keyword_comments["NEXTEND"])
                for ext in adoutput:
                    extname = ext.extname()
                    extver = ext.extver()

                    # Update AMPNAME
                    if extname=="SCI" or extname=="VAR":
                        new_ampname = ",".join(ampname[extver])

                        # These ampnames can be long, so truncate
                        # the comment by hand to avoid the error
                        # message from pyfits
                        comment = self.keyword_comments["AMPNAME"]
                        if len(new_ampname)>=65:
                            comment = ""
                        else:
                            comment = comment[0:65-len(new_ampname)]
                        ext.set_key_value("AMPNAME",new_ampname,
                                          comment=comment)

                    # Update DATASEC
                    data_shape = ext.data.shape
                    new_datasec = "[1:%i,1:%i]" % (data_shape[1],
                                                   data_shape[0])
                    ext.set_key_value("DATASEC",new_datasec,
                                      comment=self.keyword_comments["DATASEC"])

                    # Update DETSEC
                    unbin_width = data_shape[1] * ad.detector_x_bin()
                    old_detsec = refsec[extver]["DET"]
                    new_detsec = "[%i:%i,%i:%i]" % (old_detsec[0]+1,
                                              old_detsec[0]+unbin_width,
                                              old_detsec[2]+1,old_detsec[3])
                    ext.set_key_value("DETSEC",new_detsec,
                                      comment=self.keyword_comments["DETSEC"])

                    # Update CCDSEC
                    old_ccdsec = refsec[extver]["CCD"]
                    new_ccdsec = "[%i:%i,%i:%i]" % (old_ccdsec[0]+1,
                                              old_ccdsec[0]+unbin_width,
                                              old_ccdsec[2]+1,old_ccdsec[3])
                    ext.set_key_value("CCDSEC",new_ccdsec,
                                      comment=self.keyword_comments["CCDSEC"])

                    # Update CRPIX1
                    crpix1 = ext.get_key_value("CRPIX1")
                    if crpix1 is not None:
                        new_crpix1 = crpix1 + ref_shift[extver]
                        ext.set_key_value(
                            "CRPIX1",new_crpix1,
                            comment=self.keyword_comments["CRPIX1"])

                
                # Update and attach OBJCAT if needed
                if ad["OBJCAT"] is not None:
                    adoutput = _tile_objcat(ad,adoutput,mapping_dict)[0]

                # Refresh AstroData types in output file (original ones
                # were lost when new AD was created)
                adoutput.refresh_types()
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=adoutput, keyword=timestamp_key)

            # Change the filename
            adoutput.filename = gt.filename_updater(
                adinput=adoutput, suffix=rc["suffix"], strip=True)
            
            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(adoutput)
        
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
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "trimOverscan", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["trimOverscan"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the trimOverscan primitive has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by trimOverscan" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Trim the data to its data_section descriptor and update
            # keywords to match
            ad = gt.trim_to_data_section(ad)[0]
            
            # Set 'TRIMMED' to 'yes' in the PHU and update the log
            ad.phu_set_key_value("TRIMMED","yes",
                                 comment=self.keyword_comments["TRIMMED"])

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"], 
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

        :param repair: Set to True to repair the data. Note: this feature does
                       not work yet.
        :type repair: Python boolean
        """
        
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
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

            if repair:
                # Set repair to False, since it doesn't work at the moment
                log.warning("Setting repair=False, since this functionality "
                            "is not yet implemented")
                repair = False

            # Validate the input AstroData object by ensuring that it has
            # 1, 2, 3, 4, 6 or 12 extensions
            valid_num_ext = [1, 2, 3, 4, 6, 12]
            num_ext = ad.count_exts("SCI")
            if num_ext not in valid_num_ext:
                if repair:
                    # This would be where we would attempt to repair the data
                    # This shouldn't happen while repair = False exists above
                    pass
                else:
                    raise Errors.Error("The number of extensions in %s do " \
                                       "match with the number of extensions " \
                                       "expected in raw GMOS data." \
                                           % (ad.filename))
                    
            else:
                log.fullinfo("The GMOS input file has been validated: %s " \
                             "contains %d extensions" % (ad.filename, num_ext))


            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"], 
                                              strip=True)
            
            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc

##############################################################################
# Below are the helper functions for the primitives in this module           #
##############################################################################

def _obtain_arraygap(adinput=None):
    """
    This function obtains the raw array gap size for the different GMOS
    detectors and returns it after correcting for binning. There are two
    values in the GMOSArrayGaps.py file in the GMOS
    lookup directory, one for unbinned data and one to be used to calculate
    the chip gap when the data are binned.
    """
    
    # Get the dictionary containing the CCD gaps
    all_arraygaps_dict = Lookups.get_lookup_table(\
        "Gemini/GMOS/GMOSArrayGaps.py","gmosArrayGaps")
    
    # Obtain the X binning and detector type for the ad input
    detector_x_bin = adinput.detector_x_bin()
    detector_type = adinput.phu_get_key_value("DETTYPE")

    # Check the read values
    if detector_x_bin is None or detector_type is None:
        if hasattr(ad, "exception_info"):
            raise adinput.exception_info
    
    # Check if the data are binned
    if detector_x_bin > 1:
        bin_string = "binned"
    else:
        bin_string = "unbinned"

    # Form the key
    key = (detector_type, bin_string)

    # Obtain the array gap value and fix for any binning
    if key in all_arraygaps_dict:
        arraygap = all_arraygaps_dict[key] / detector_x_bin.as_pytype()
    else:
        raise Errors.ScienceError("Array gap value not " +
                                  "found for %s" % (detector_type)) 
    return arraygap

def _tile_objcat(adinput=None,adoutput=None,mapping_dict=None):
    """
    This function tiles together separate OBJCAT extensions, converting
    the pixel coordinates to the new WCS.
    """

    adinput = gt.validate_input(adinput=adinput)
    adoutput = gt.validate_input(adinput=adoutput)

    if mapping_dict is None:
        raise Errors.InputError("mapping_dict must not be None")

    if len(adinput)!=len(adoutput):
        raise Errors.InputError("adinput must have same length as adoutput")
    output_dict = gt.make_dict(key_list=adinput, value_list=adoutput)

    adoutput_list = []
    for ad in adinput:
        
        adout = output_dict[ad]

        objcat = ad["OBJCAT"]
        if objcat is None:
            raise Errors.InputError("No OBJCAT found in %s" % ad.filename)

        for outext in adout["SCI"]:
            out_extver = outext.extver()
            output_wcs = pywcs.WCS(outext.header)

            col_names = None
            col_fmts = None
            col_data = {}
            for inp_extver in mapping_dict[out_extver]:
                inp_objcat = ad["OBJCAT",inp_extver]

                # Make sure there is data in the OBJCAT
                if inp_objcat is None:
                    continue
                if inp_objcat.data is None:
                    continue
                if len(inp_objcat.data)==0:
                    continue

                # Get column names, formats from first OBJCAT
                if col_names is None:
                    col_names = inp_objcat.data.names
                    col_fmts = inp_objcat.data.formats
                    for name in col_names:
                        col_data[name] = inp_objcat.data.field(name).tolist()
                else:
                    # Stack all OBJCAT data together
                    for name in col_names:
                        col_data[name].extend(inp_objcat.data.field(name))

            # Get new pixel coordinates for the objects from RA/Dec
            # and the output WCS
            ra = col_data["X_WORLD"]
            dec = col_data["Y_WORLD"]
            newx,newy = output_wcs.wcs_sky2pix(ra,dec,1)
            col_data["X_IMAGE"] = newx
            col_data["Y_IMAGE"] = newy

            columns = {}
            for name,format in zip(col_names,col_fmts):
                # Let add_objcat auto-number sources
                if name=="NUMBER":
                    continue

                # Define pyfits column to pass to add_objcat
                columns[name] = pf.Column(name=name,format=format,
                                          array=col_data[name])

            adout = gt.add_objcat(adinput=adout, extver=out_extver,
                                  columns=columns)[0]

        adoutput_list.append(adout)

    return adoutput_list
