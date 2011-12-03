import os
import numpy as np
from astrodata import AstroData
from astrodata import Errors
from astrodata import IDFactory
from astrodata.adutils import gemLog
from astrodata.adutils.reduceutils.prsproxyutil import upload_calibration
from gempy import geminiTools as gt
from primitives_GENERAL import GENERALPrimitives

class PreprocessingPrimitives(GENERALPrimitives):
    """
    This is the class containing all of the preprocessing primitives
    for the GEMINI level of the type hierarchy tree. It inherits all
    the primitives from the level above, 'GENERALPrimitives'.
    """
    astrotype = "GEMINI"
    
    def init(self, rc):
        GENERALPrimitives.init(self, rc)
        return rc
    init.pt_hide = True
    
    def aduToElectrons(self, rc):
        """
        This primitive will convert the units of the pixel data extensions
        of the input AstroData object from ADU to electrons by multiplying
        by the gain.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "aduToElectrons", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["aduToElectrons"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the aduToElectrons primitive has been run
            # previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by aduToElectrons" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Convert the pixel data in the AstroData object from ADU to
            # electrons. First, get the gain value using the appropriate
            # descriptor
            gain = ad.gain()

            # Now multiply the pixel data in the science extension by the gain
            # and the pixel data in the variance extension by the gain squared
            log.fullinfo("Converting %s from ADU to electrons by " \
                         "multiplying the science extension by the gain = %s" \
                         % (ad.filename, gain))
            ad = ad.mult(gain)
            
            # Update the headers of the AstroData Object. The pixel data now
            # has units of electrons so update the physical units keyword.
            for ext in ad["SCI"]:
                ext.set_key_value("BUNIT","electron",
                                  comment=self.keyword_comments["BUNIT"])
            varext = ad["VAR"]
            if varext is not None:
                for ext in varext:
                    ext.set_key_value("BUNIT","electron*electron",
                                      comment=self.keyword_comments["BUNIT"])

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
    
    def correctBackgroundToReferenceImage(self, rc):
        """
        This primitive does an additive correction to a set
        of images to put their sky background at the same level
        as the reference image before stacking.
        """

        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", 
                                 "correctBackgroundToReferenceImage",
                                 "starting"))

        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["correctBackgroundToReferenceImage"]

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Get input files
        adinput = rc.get_inputs_as_astrodata()
        
        # Get the number of science extensions in each file
        next = np.array([ad.count_exts("SCI") for ad in adinput])

        # Initialize reference BG
        ref_bg = None

        # Check whether two or more input AstroData objects were provided
        if len(adinput) <= 1:
            log.warning("No correction will be performed, since at least " \
                        "two input AstroData objects are required for " \
                        "correctBackgroundToReferenceImage")

            # Set the input AstroData object list equal to the output AstroData
            # objects list without further processing
            adoutput_list = adinput

        # Check that all images have the same number of science extensions
        elif not np.all(next==next[0]):
            raise Errors.InputError("Number of science extensions in input "\
                                    "images do not match")
        else:

            # Check whether measureBG needs to be run
            bg_list = [sciext.get_key_value("SKYLEVEL") \
                           for ad in adinput for sciext in ad["SCI"]]
            if None in bg_list:
                log.fullinfo("SKYLEVEL not found, measuring background")
                rc.run("measureBG(separate_ext=True)")

            # Loop over input files
            for ad in adinput:
                ref_bg_dict = {}
                diff_dict = {}
                for sciext in ad["SCI"]:
                    # Get background value from header
                    bg = sciext.get_key_value("SKYLEVEL")
                    if bg is None:
                        raise Errors.ScienceError(
                            "Could not get background level from %s[SCI,%d]" %
                            (sciext.filename,sciext.extver))
                    
                    log.fullinfo("Background level is %.0f for %s" %
                                 (bg, ad.filename))
                    if ref_bg is None:
                        ref_bg_dict[(sciext.extname(),sciext.extver())]=bg
                    else:
                        ref = ref_bg[(sciext.extname(),sciext.extver())]
                        difference = ref - bg
                        log.fullinfo("Adding %.0f to match reference " \
                                     "background level %.0f" % 
                                     (difference,ref))
                        sciext.add(difference)
                        sciext.set_key_value(
                            "SKYLEVEL",bg+difference,
                            comment=self.keyword_comments["SKYLEVEL"])

                # Store background level of first image
                if ref_bg is None:
                    ref_bg = ref_bg_dict

                # Add time stamps, change the filename, and
                # append to output list
                gt.mark_history(adinput=ad, keyword=timestamp_key)
                ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                                 strip=True)
                adoutput_list.append(ad)
 
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)

        yield rc

    def divideByFlat(self, rc):
        """
        This primitive will divide each SCI extension of the inputs by those
        of the corresponding flat. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the division on the data.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "divideByFlat", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["divideByFlat"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the divideByFlat primitive has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by divideByFlat" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Retrieve the appropriate flat from the reduction context
            flat = AstroData(rc.get_cal(ad, "processed_flat"))
            
            # If there is no appropriate flat, there is no need to divide by
            # the flat in QA context; in SQ context, raise an error
            if flat.filename is None:
                if "qa" in rc.context:
                    log.warning("No changes will be made to %s, since no " \
                                "appropriate flat could be retrieved" \
                                % (ad.filename))
                    # Append the input AstroData object to the list of output
                    # AstroData objects without further processing
                    adoutput_list.append(ad)
                    continue
                else:
                    raise Errors.PrimitiveError("No processed flat found "\
                                                "for %s" % ad.filename)
            
            # Check the inputs have matching filters, binning, and SCI shapes.
            try:
                gt.checkInputsMatch(adInsA=ad, adInsB=flat) 
            except Errors.ToolboxError:
                # If not, try to clip the flat frame to the size
                # of the science data
                # For a GMOS example, this allows a full frame flat to
                # be used for a CCD2-only science frame. 
                flat = gt.clip_auxiliary_data(
                    adinput=ad,aux=flat,aux_type="cal")[0]

                # Check again, but allow it to fail if they still don't match
                gt.checkInputsMatch(adInsA=ad, adInsB=flat)


            # Divide the adinput by the flat
            log.fullinfo("Dividing the input AstroData object (%s) " \
                         "by this flat:\n%s" % (ad.filename,
                                                flat.filename))
            ad = ad.div(flat)
            
            # Record the flat file used
            ad.phu_set_key_value("FLATIM", 
                                 os.path.basename(flat.filename),
                                 comment=self.keyword_comments["FLATIM"])

            
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
     
    def failCalibration(self,rc):
        # Mark a given calibration "fail" and upload it 
        # to the system. This is intended to be used to mark a 
        # calibration file that has already been uploaded, so that
        # it will not be returned as a valid match for future data.
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Initialize the list of output AstroData objects 
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Change the two keywords -- BAD and NO = Fail
            ad.phu_set_key_value("RAWGEMQA","BAD",
                                 comment=self.keyword_comments["RAWGEMQA"])
            ad.phu_set_key_value("RAWPIREQ","NO",
                                 comment=self.keyword_comments["RAWPIREQ"])
            log.fullinfo("%s has been marked %s" % (ad.filename,ad.qa_state()))
            
            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the 
        # reduction context
        rc.report_output(adoutput_list)
        
        # Run the storeCalibration primitive, so that the 
        # failed file gets re-uploaded
        rc.run("storeCalibration")
        
        yield rc
            
    def getCalibration(self, rc):
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Retrieve type of calibration requested
        caltype = rc["caltype"]
        if caltype == None:
            log.error("getCalibration: caltype not set")
            raise Errors.PrimitiveError("getCalibration: caltype not set")
        
        # Retrieve source of calibration
        source = rc["source"]
        if source == None:
            source = "all"
            
        # Check whether calibrations are already available
        calibrationless_adlist = []
        adinput = rc.get_inputs_as_astrodata()
        #for ad in adinput:
        #    ad.mode = "update"
        #    calurl = rc.get_cal(ad,caltype)
        #    if not calurl:
        #        calibrationless_adlist.append(ad)
        calibrationless_adlist = adinput
        # Request any needed calibrations
        if len(calibrationless_adlist) ==0:
            # print "pG603: calibrations for all files already present"
            pass
        else:
            rc.rq_cal(caltype, calibrationless_adlist, source=source)
        
        yield rc
        
    def getProcessedBias(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        caltype = "processed_bias"
        source = rc["source"]
        if source == None:
            rc.run("getCalibration(caltype=%s)" % caltype)
        else:
            rc.run("getCalibration(caltype=%s, source=%s)" % (caltype,source))
        
        # List calibrations found
        first = True
        for ad in rc.get_inputs_as_astrodata():
            calurl = rc.get_cal(ad, caltype) #get from cache
            if calurl:
                cal = AstroData(calurl)
                if cal.filename is None:
                    if "qa" not in rc.context:
                        raise Errors.InputError("Calibration not found for " \
                                                "%s" % ad.filename)
                else:
                    if first:
                        log.stdinfo("getCalibration: Results")
                        first = False
                    log.stdinfo("   %s\n      for %s" % (cal.filename,
                                                         ad.filename))
            else: 
                if "qa" not in rc.context:
                    raise Errors.InputError("Calibration not found for %s" % 
                                            ad.filename)
        
        yield rc
    
    def getProcessedDark(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        caltype = "processed_dark"
        source = rc["source"]
        if source == None:
            rc.run("getCalibration(caltype=%s)" % caltype)
        else:
            rc.run("getCalibration(caltype=%s, source=%s)" % (caltype,source))
            
        # List calibrations found
        first = True
        for ad in rc.get_inputs_as_astrodata():
            calurl = rc.get_cal(ad, caltype) #get from cache
            if calurl:
                cal = AstroData(calurl)
                if cal.filename is None:
                    if "qa" not in rc.context:
                        raise Errors.InputError("Calibration not found for " \
                                                "%s" % ad.filename)
                else:
                    if first:
                        log.stdinfo("getCalibration: Results")
                        first = False
                    log.stdinfo("   %s\n      for %s" % (cal.filename,
                                                         ad.filename))
            else: 
                if "qa" not in rc.context:
                    raise Errors.InputError("Calibration not found for %s" % 
                                            ad.filename)
        
        yield rc
    
    def getProcessedFlat(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        caltype = "processed_flat"
        source = rc["source"]
        if source == None:
            rc.run("getCalibration(caltype=%s)" % caltype)
        else:
            rc.run("getCalibration(caltype=%s, source=%s)" % (caltype,source))
        
        # List calibrations found
        first = True
        for ad in rc.get_inputs_as_astrodata():
            calurl = rc.get_cal(ad, caltype) #get from cache
            if calurl:
                cal = AstroData(calurl)
                if cal.filename is None:
                    if "qa" not in rc.context:
                        raise Errors.InputError("Calibration not found for " \
                                                "%s" % ad.filename)
                else:
                    if first:
                        log.stdinfo("getCalibration: Results")
                        first = False
                    log.stdinfo("   %s\n      for %s" % (cal.filename,
                                                         ad.filename))
            else: 
                if "qa" not in rc.context:
                    raise Errors.InputError("Calibration not found for %s" % 
                                            ad.filename)
        
        yield rc
    
    def nonlinearityCorrect(self, rc):
        """
        Run on raw or nprepared Gemini NIRI data, this script calculates and
        applies a per-pixel linearity correction based on the counts in the
        pixel, the exposure time, the read mode, the bias level and the ROI.
        Pixels over the maximum correctable value are set to BADVAL unless
        given the force flag. Note that you may use glob expansion in infile,
        however, any pattern matching characters (*,?) must be either quoted
        or escaped with a backslash. Do we need a badval parameter that defines
        a value to assign to uncorrectable pixels, or do we want to just add
        those pixels to the DQ plane with a specific value?
        """
          
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "nonlinearityCorrect",
                                "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["nonlinearityCorrect"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Define a lookup dictionary for necessary values
        # This should be moved to the lookups area
        lincorlookup = {
            # In the following form for NIRI data:
            #("read_mode", naxis2, "well_depth_setting"):
            #    (maximum counts, exposure time correction, gamma, eta)
            ("Low Background", 1024, "Shallow"):
                (12000, 1.2662732, 7.3877618e-06, 1.940645271e-10),
            ("Medium Background", 1024, "Shallow"):
                (12000, 0.09442515154, 3.428783846e-06, 4.808353308e-10),
            ("Medium Background", 256, "Shallow"):
                (12000, 0.01029262589, 6.815415667e-06, 2.125210479e-10),
            ("High Background", 1024, "Shallow"):
                (12000, 0.009697324059, 3.040036696e-06, 4.640788333e-10),
            ("High Background", 1024, "Deep"):
                (21000, 0.007680816203, 3.581914163e-06, 1.820403678e-10),
            }

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the nonlinearityCorrect primitive has been run
            # previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by nonlinearityCorrect" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue

            # Get the appropriate information using the descriptors
            coadds = ad.coadds()
            read_mode = ad.read_mode().as_pytype()
            total_exposure_time = ad.exposure_time()
            well_depth_setting = ad.well_depth_setting().as_pytype()
            if coadds is None or read_mode is None or \
                total_exposure_time is None or well_depth_setting is None:
                # The descriptor functions return None if a value cannot be
                # found and stores the exception info. Re-raise the
                # exception.
                if hasattr(ad, "exception_info"):
                    raise ad.exception_info
            
            # Check the raw exposure time (i.e., per coadd). First, convert
            # the total exposure time returned by the descriptor back to
            # the raw exposure time
            exposure_time = total_exposure_time / coadds
            if exposure_time > 600.:
                log.critical("The raw exposure time is outside the " \
                             "range used to derive correction.")
                raise Errors.InvalidValueError()
            
            # Check the read mode and well depth setting values
            if read_mode == "Invalid" or well_depth_setting == "Invalid":
                raise Errors.CalcError()
            
            # Print the descriptor values
            log.fullinfo("The number of coadds = %s" % coadds)
            log.fullinfo("The read mode = %s" % read_mode)
            log.fullinfo("The total exposure time = %s" % total_exposure_time)
            log.fullinfo("The well depth = %s" % well_depth_setting)
            
            # Loop over each science extension in each input AstroData object
            for ext in ad["SCI"]:
                
                # Get the size of the raw pixel data
                naxis2 = ext.get_key_value("NAXIS2")
                
                # Get the raw pixel data
                raw_pixel_data = ext.data
                
                # Divide the raw pixel data by the number of coadds
                if coadds > 1:
                    raw_pixel_data = raw_pixel_data / coadds
                
                # Determine the mean of the raw pixel data
                raw_mean_value = np.mean(raw_pixel_data)
                log.fullinfo("The mean value of the raw pixel data in " \
                             "%s is %.8f" % (ext.filename, raw_mean_value))
                
                # Create the key used to access the coefficients that are
                # used to correct for non-linearity
                key = (read_mode, naxis2, well_depth_setting)
                
                # Get the coefficients from the lookup table
                if lincorlookup[key]:
                    maximum_counts, coeff1, coeff2, coeff3 = \
                        lincorlookup[key]
                else:
                    raise Errors.TableKeyError()
                log.fullinfo("Coefficients used = %.12f, %.9e, %.9e" \
                             % (coeff1, coeff2, coeff3))
                
                # Create a new array that contains the corrected pixel data
                corrected_pixel_data = raw_pixel_data + \
                    coeff2 * raw_pixel_data**2 + coeff3 * raw_pixel_data**3
                
                # nirlin replaces pixels greater than maximum_counts with 0
                # Set the pixels to 0 if they have a value greater than the
                # maximum counts
                #log.fullinfo("Setting pixels to zero if above %f" % \
                #    maximum_counts)
                #corrected_pixel_data[corrected_pixel_data > \
                # maximum_counts] = 0
                # Should probably add the above to the DQ plane
                
                # Multiply the corrected pixel data by the number of coadds
                if coadds > 1:
                    corrected_pixel_data = corrected_pixel_data * coadds
                
                # Write the corrected pixel data to the output object
                ext.data = corrected_pixel_data
                
                # Determine the mean of the corrected pixel data
                corrected_mean_value = np.mean(ext.data)
                log.fullinfo("The mean value of the corrected pixel data in " \
                             "%s is %.8f" \
                             % (ext.filename, corrected_mean_value))
            
            # Correct the exposure time by adding coeff1
            total_exposure_time = total_exposure_time + coeff1
            log.fullinfo("The corrected total exposure time = %f" \
                         % total_exposure_time)
            
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
    
    def showCals(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        if str(rc["showcals"]).lower() == "all":
            num = 0
            # print "pG256: showcals=all", repr (rc.calibrations)
            for calkey in rc.calibrations:
                num += 1
                log.stdinfo(rc.calibrations[calkey], category="calibrations")
            if (num == 0):
                log.stdinfo("There are no calibrations in the cache.")
        else:
            for adr in rc.inputs:
                sid = IDFactory.generate_astro_data_id(adr.ad)
                num = 0
                for calkey in rc.calibrations:
                    if sid in calkey :
                        num += 1
                        log.stdinfo(rc.calibrations[calkey], 
                                     category="calibrations")
            if (num == 0):
                log.stdinfo("There are no calibrations in the cache.")
        
        yield rc
    ptusage_showCals="Used to show calibrations currently in cache for inputs."
    
    def storeCalibration(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # print repr(rc)
        storedcals = rc["cachedict"]["storedcals"]
        
        for ad in rc.get_inputs_as_astrodata():
            fname = os.path.join(storedcals, os.path.basename(ad.filename))
            ad.filename = fname
            # always clobber, perhaps save clobbered file somewhere
            ad.write(filename = fname, rename = True, clobber=True)
            log.stdinfo("File saved to %s" % fname)
            if "upload" in rc.context:
                try:
                    upload_calibration(ad.filename)
                except:
                    log.warning("Unable to upload file to calibration system")
                else:
                    log.stdinfo("File stored in calibration system")
            yield rc
        
        yield rc
    
    def storeProcessedBias(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedBias",
                                 "starting"))
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"],
                                             strip=True)
            
            # Adding a PROCBIAS time stamp to the PHU
            gt.mark_history(adinput=ad, keyword="PROCBIAS")
            
            # Refresh the AD types to reflect new processed status
            ad.refresh_types()
        
        # Upload bias(es) to cal system
        rc.run("storeCalibration")
        
        yield rc
    
    def storeProcessedDark(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedDark",
                                 "starting"))
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"],
                                             strip=True)
            
            # Adding a PROCDARK time stamp to the PHU
            gt.mark_history(adinput=ad, keyword="PROCDARK")
            
            # Refresh the AD types to reflect new processed status
            ad.refresh_types()
        
        # Upload to cal system
        rc.run("storeCalibration")
        
        yield rc
    
    def storeProcessedFlat(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedFlat",
                                 "starting"))
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"],
                                             strip=True)
            
            # Adding a PROCFLAT time stamp to the PHU
            gt.mark_history(adinput=ad, keyword="PROCFLAT")
            
            # Refresh the AD types to reflect new processed status
            ad.refresh_types()
        
        # Upload to cal system
        rc.run("storeCalibration")
        
        yield rc
    
    def subtractDark(self, rc):
        """
        This primitive will subtract each SCI extension of the inputs by those
        of the corresponding dark. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the subtraction on the 
        data.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "subtractDark", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["subtractDark"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the subtractDark primitive has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by subtractDark" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Retrieve the appropriate dark from the reduction context
            dark = AstroData(rc.get_cal(ad, "processed_dark"))

            # If there is no appropriate dark, there is no need to
            # subtract the dark
            if dark.filename is None:
                log.warning("No changes will be made to %s, since no " \
                            "appropriate dark could be retrieved" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Subtract the dark from the input AstroData object
            log.fullinfo("Subtracting the dark (%s) from the input " \
                         "AstroData object %s" \
                         % (dark.filename, ad.filename))
            ad = ad.sub(dark)
            
            # Record the dark file used
            ad.phu_set_key_value("DARKIM", 
                                 os.path.basename(dark.filename),
                                 comment=self.keyword_comments["DARKIM"])

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

