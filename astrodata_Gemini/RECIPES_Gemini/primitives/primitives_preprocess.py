import os
from copy import deepcopy
import datetime
import math
import numpy as np
from astrodata import AstroData
from astrodata import Errors
from astrodata import ReductionContextRecords as RCR
from astrodata.adutils import logutils
from gempy.gemini import gemini_tools as gt
from primitives_GENERAL import GENERALPrimitives

class PreprocessPrimitives(GENERALPrimitives):
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
        log = logutils.get_logger(__name__)
        
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
                log.warning("No changes will be made to %s, since it has "
                            "already been processed by aduToElectrons"
                            % ad.filename)
                
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
            log.status("Converting %s from ADU to electrons by multiplying "
                       "the science extension by the gain = %s"
                       % (ad.filename, gain))
            ad = ad.mult(gain)
            
            # Update the headers of the AstroData Object. The pixel data now
            # has units of electrons so update the physical units keyword.
            gt.update_key(adinput=ad, keyword="BUNIT", value="electron",
                          comment=None, extname="SCI")
            if ad["VAR"]:
                gt.update_key(adinput=ad, keyword="BUNIT",
                              value="electron*electron", comment=None,
                              extname="VAR")
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            
            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"], 
                                              strip=True)
            
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def associateSky(self, rc):
        """
        This primitive determines which sky AstroData objects are associated
        with each science AstroData object and adds this information to a
        dictionary (in the form {science1:[sky1,sky2],science2:[sky2,sky3]}),
        where science1 and science2 are the science AstroData objects and sky1,
        sky2 and sky3 are the sky AstroDataRecord objects, which is then added
        to the reduction context.
        
        The input sky AstroData objects can be provided by the user using the
        parameter 'sky'. Otherwise, the science AstroData objects are found in
        the main stream (as normal) and the sky AstroData objects are found in
        the sky stream.
        
        :param adinput: input science AstroData objects
        :type adinput: Astrodata or Python list of AstroData
        
        :param sky: The input sky frame(s) to be subtracted from the input
                    science frame(s). The input sky frame(s) can be a list of
                    sky filenames, a sky filename, a list of AstroData objects
                    or a single AstroData object. Note: If there are multiple
                    input science frames and one input sky frame provided, then
                    the same sky frame will be applied to all inputs; otherwise
                    the number of input sky frames must match the number of
                    input science frames.
        :type sky: string, Python list of string, AstroData or Python list of
                   Astrodata 
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "associateSky", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["associateSky"]
        
        # Determine the suffix for this primitive
        suffix = rc["suffix"]
        
        # This primitives requires at least one input science AstroData object
        # and at least one input sky AstroData object
        ad_science_list = rc.get_inputs_as_astrodata()
        
        if rc["sky"]:
            # Use the list of sky frames provided by the user
            ad_sky_list = []
            for sky in rc["sky"]:
                if not isinstance(sky, AstroData):
                    sky = AstroData(sky)
                
                # Create a list of sky AstroDataRecord objects
                ad_sky_list.append(sky)
        else:
            # The seperateSky primitive puts the sky AstroData objects in the
            # sky stream. The get_stream function returns a list of AstroData
            # objects when style="AD"
            ad_sky_list = rc.get_stream(stream="sky", style="AD")
        
        if not ad_science_list or not ad_sky_list:
            log.warning("Cannot associate sky frames, since at least one "
                        "science AstroData object and one sky AstroData "
                        "object are required for associateSky")
            
            # Add the science and sky AstroData objects to the output science
            # and sky AstroData object lists, respectively, without further
            # processing, after adding the appropriate time stamp to the PHU
            # and updating the filename.
            if ad_science_list:
                ad_science_output_list = gt.finalise_adinput(
                  adinput=ad_science_list, timestamp_key=timestamp_key,
                  suffix=suffix)
            
            if ad_sky_list:
                ad_sky_output_list = gt.finalise_adinput(
                  adinput=ad_sky_list, timestamp_key=timestamp_key,
                  suffix=suffix)
        else:
            # Initialize the dictionary that will contain the association
            # between the science AstroData objects and the sky AstroData
            # objects 
            sky_dict = {}
            
            # Loop over each science AstroData object in the science list
            for ad_science in ad_science_list:
                
                # Determine the sky AstroData objects that are associated with
                # this science AstroData object. Initialize the list of sky
                # AstroDataRecord objects
                adr_sky_list = []
                
                # Use the ORIGNAME of the science AstroData object as the key
                # of the dictionary 
                origname = ad_science.phu_get_key_value("ORIGNAME")
                
                # If use_all is True, use all of the sky AstroData objects for
                # each science AstroData object
                if rc["use_all"]:
                    log.stdinfo("Associating all available sky AstroData "
                                 "objects to %s" % ad_science.filename)
                    
                    # Set the list of sky AstroDataRecord objects for this
                    # science AstroData object equal to the input list of sky
                    # AstroDataRecord objects
                    adr_sky_list = RCR.AstroDataRecord(ad_sky_list)
                    
                    # Update the dictionary with the list of sky
                    # AstroDataRecord objects associated with this science
                    # AstroData object
                    sky_dict.update({origname: adr_sky_list})
                else:
                    # Get the datetime object of the science AstroData object
                    # using the appropriate descriptor 
                    ad_science_datetime = ad_science.ut_datetime()
                    
                    # Loop over each sky AstroData object in the sky list
                    for ad_sky in ad_sky_list:
                    
                        # Get the datetime object of the sky AstroData object
                        # using the appropriate descriptor
                        ad_sky_datetime = ad_sky.ut_datetime()
                        
                        # Create a timedelta object using the value of the
                        # "time" parameter
                        seconds = datetime.timedelta(seconds=rc["time"])
                        
                        # Select only those sky AstroData objects observed
                        # within "time" seconds of the science AstroData object
                        if (abs(ad_science_datetime - ad_sky_datetime) <
                            seconds):
                            
                            # Get the distance of the science and sky AstroData
                            # objects using the x_offset and y_offset
                            # descriptors
                            ad_science_distance = math.sqrt(
                              ad_science.x_offset()**2 +
                              ad_science.y_offset()**2)
                            
                            ad_sky_distance = math.sqrt(
                              ad_sky.x_offset()**2 + ad_sky.y_offset()**2)
                            
                            # Select only those sky AstroData objects that are
                            # greater than "distance" arcsec away from the
                            # science AstroData object
                            if (abs(ad_science_distance - ad_sky_distance) >
                                rc["distance"]):
                                adr_sky_list.append(
                                  RCR.AstroDataRecord(ad_sky))
                    
                    # Update the dictionary with the list of sky
                    # AstroDataRecord objects associated with this science
                    # AstroData object
                    sky_dict.update({origname: adr_sky_list})
                
                if not sky_dict[origname]:
                    log.warning("No sky frames available for %s" % origname)
                else:
                    log.stdinfo("The sky frames associated with %s are:"
                                 % origname)
                    for adr_sky in sky_dict[origname]:
                        log.stdinfo("  %s" % adr_sky.ad.filename)
            
            # Add the appropriate time stamp to the PHU and change the filename
            # of the science and sky AstroData objects 
            ad_science_output_list = gt.finalise_adinput(
              adinput=ad_science_list, timestamp_key=timestamp_key,
              suffix=suffix)
            
            ad_sky_output_list = gt.finalise_adinput(
              adinput=ad_sky_list, timestamp_key=timestamp_key, suffix=suffix)
            
            # Add the association dictionary to the reduction context
            rc["sky_dict"] = sky_dict
        
        # Report the list of output sky AstroData objects to the sky stream in
        # the reduction context 
        rc.report_output(ad_sky_output_list, stream="sky")
        
        # Report the list of output science AstroData objects to the reduction
        # context 
        rc.report_output(ad_science_output_list)
        
        yield rc
    
    def correctBackgroundToReferenceImage(self, rc):
        """
        This primitive does an additive correction to a set
        of images to put their sky background at the same level
        as the reference image before stacking.
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)

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

        # Check whether we are scaling to zero or to 1st image
        remove_zero_level = rc["remove_zero_level"]

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
                if rc["logLevel"]=="stdinfo":
                    log.changeLevels(logLevel="status")
                    rc.run("measureBG(separate_ext=True,remove_bias=False)")
                    log.changeLevels(logLevel=rc["logLevel"])
                else:
                    rc.run("measureBG(separate_ext=True,remove_bias=False)")

            # Loop over input files
            for ad in adinput:
                ref_bg_dict = {}
                diff_dict = {}
                for sciext in ad["SCI"]:
                    # Get background value from header
                    bg = sciext.get_key_value("SKYLEVEL")
                    if bg is None:
                        if "qa" in rc.context:
                            log.warning(
                                "Could not get background level from "\
                                "%s[SCI,%d]" %
                                (sciext.filename,sciext.extver()))
                            continue
                        else:
                            raise Errors.ScienceError(
                                "Could not get background level from "\
                                "%s[SCI,%d]" %
                                (sciext.filename,sciext.extver()))
                    
                    log.fullinfo("Background level is %.0f for %s" %
                                 (bg, ad.filename))
                    if ref_bg is None:
                        if remove_zero_level:
                            log.fullinfo("Subtracting %.0f to remove " \
                                         "zero level from reference image" %
                                         bg)
                            sciext.sub(bg)
                            ref_bg_dict[(sciext.extname(),sciext.extver())]=0
                        else:
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
                if ref_bg is None and ref_bg_dict:
                    ref_bg = ref_bg_dict

                # Add time stamps, change the filename, and
                # append to output list
                gt.mark_history(adinput=ad, keyword=timestamp_key)
                ad.filename = gt.filename_updater(adinput=ad, 
                                                  suffix=rc["suffix"], 
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
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "divideByFlat", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["divideByFlat"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Check for a user-supplied flat
        adinput = rc.get_inputs_as_astrodata()
        flat_param = rc["flat"]
        flat_dict = None
        if flat_param is not None:
            # The user supplied an input to the flat parameter
            if not isinstance(flat_param, list):
                flat_list = [flat_param]
            else:
                flat_list = flat_param

            # Convert filenames to AD instances if necessary
            tmp_list = []
            for flat in flat_list:
                if type(flat) is not AstroData:
                    flat = AstroData(flat)
                tmp_list.append(flat)
            flat_list = tmp_list
            
            flat_dict = gt.make_dict(key_list=adinput, value_list=flat_list)

        # Loop over each input AstroData object in the input list
        for ad in adinput:
            
            # Check whether the divideByFlat primitive has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by divideByFlat" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Retrieve the appropriate flat
            if flat_dict is not None:
                flat = flat_dict[ad]
            else:
                flat = rc.get_cal(ad, "processed_flat")
            
                # If there is no appropriate flat, there is no need to divide by
                # the flat in QA context; in SQ context, raise an error
                if flat is None:
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
                else:
                    flat = AstroData(flat)
            
            # Check the inputs have matching filters, binning, and SCI shapes.
            try:
                gt.check_inputs_match(ad1=ad, ad2=flat) 
            except Errors.ToolboxError:
                # If not, try to clip the flat frame to the size
                # of the science data
                # For a GMOS example, this allows a full frame flat to
                # be used for a CCD2-only science frame. 
                flat = gt.clip_auxiliary_data(
                    adinput=ad,aux=flat,aux_type="cal")[0]

                # Check again, but allow it to fail if they still don't match
                gt.check_inputs_match(ad1=ad, ad2=flat)


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
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"], 
                                              strip=True)
            
            # Append the output AstroData object to the list 
            # of output AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
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
        log = logutils.get_logger(__name__)
        
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
            coadds = ad.coadds().as_pytype()
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
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"], 
                                              strip=True)
            
            # Append the output AstroData object to the list 
            # of output AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def normalizeFlat(self, rc):
        """
        This primitive normalizes each science extension of the input
        AstroData object by its mean
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "normalizeFlat", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["normalizeFlat"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the normalizeFlat primitive has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by normalizeFlat" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Loop over each science extension in each input AstroData object
            for ext in ad["SCI"]:
                
                # Normalise the input AstroData object. Calculate the mean
                # value of the science extension
                mean = np.mean(ext.data)
                # Divide the science extension by the mean value of the science
                # extension
                log.fullinfo("Normalizing %s[%s,%d] by dividing by the mean " \
                             "= %f" % (ad.filename, ext.extname(),
                                       ext.extver(), mean))
                ext = ext.div(mean)

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

    def separateSky(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "separateSky", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["separateSky"]
        
        # Determine the suffix for this primitive
        suffix = rc["suffix"]
        
        # This primitive requires at least two input AstroData objects
        ad_input_list = rc.get_inputs_as_astrodata()
        
        if len(ad_input_list) < 2:
            log.warning("Cannot separate sky frames, since at least two "
                        "input AstroData objects are required for "
                        "separateSky")
            
            # Add the input AstroData objects to the output science AstroData
            # object list without further processing, after adding the
            # appropriate time stamp to the PHU and updating the filename.
            if ad_input_list:
                ad_science_output_list = gt.finalise_adinput(
                  adinput=ad_input_list, timestamp_key=timestamp_key,
                  suffix=suffix)
        else:
            # Initialize the lists of science and sky AstroData objects
            ad_science_list = []
            ad_sky_list = []
            
            # Loop over each input AstroData object in the input list
            for ad in ad_input_list:
                
                # If the all_on_source parameter is equal to yes, all the input
                # on-source AstroData objects can be used as a sky frame
                if rc["all_on_source"]:
                    ad_science_list.append(ad)
                    adsky = deepcopy(ad)
                    ad_sky_list.append(adsky)
                
                # If any of the input AstroData objects contain a "SKYFRAME"
                # keyword, that input can be used as a sky frame
                elif ad.phu_get_key_value("SKYFRAME"):
                    log.fullinfo("%s can be used as a sky frame" % ad.filename)
                    
                    # Append the input AstroData object to the list of sky
                    # AstroData objects
                    ad_sky_list.append(ad)
                
                else:
                    # Automatically determine the sky frames. For now, assume
                    # everything else is science. 
                    
                    # Append the input AstroData object to the list of science
                    # AstroData objects
                    ad_science_list.append(ad)

            log.stdinfo("Science frames:")
            for ad_science in ad_science_list:
                log.stdinfo("  %s" % ad_science.filename)

            log.stdinfo("Sky frames:")
            for ad_sky in ad_sky_list:
                log.stdinfo("  %s" % ad_sky.filename)

            # Add the appropriate time stamp to the PHU and update the filename
            # of the science and sky AstroData objects 
            ad_science_output_list = gt.finalise_adinput(
              adinput=ad_science_list, timestamp_key=timestamp_key,
              suffix=suffix)
            
            ad_sky_output_list = gt.finalise_adinput(
              adinput=ad_sky_list, timestamp_key=timestamp_key, suffix=suffix)
               
        # Report the list of output sky AstroData objects to the sky stream in
        # the reduction context
        rc.report_output(ad_sky_output_list, stream="sky")
        
        # Report the list of output science AstroData objects to the reduction
        # context
        rc.report_output(ad_science_output_list)
        
        yield rc
    
    def subtractDark(self, rc):
        """
        This primitive will subtract each SCI extension of the inputs by those
        of the corresponding dark. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the subtraction on the 
        data.
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "subtractDark", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["subtractDark"]
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Check for a user-supplied dark
        adinput = rc.get_inputs_as_astrodata()
        dark_param = rc["dark"]
        dark_dict = None
        if dark_param is not None:
            # The user supplied an input to the dark parameter
            if not isinstance(dark_param, list):
                dark_list = [dark_param]
            else:
                dark_list = dark_param

            # Convert filenames to AD instances if necessary
            tmp_list = []
            for dark in dark_list:
                if type(dark) is not AstroData:
                    dark = AstroData(dark)
                tmp_list.append(dark)
            dark_list = tmp_list
            
            dark_dict = gt.make_dict(key_list=adinput, value_list=dark_list)

        # Loop over each input AstroData object in the input list
        for ad in adinput:
            
            # Check whether the subtractDark primitive has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by subtractDark" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Retrieve the appropriate dark
            if dark_dict is not None:
                dark = dark_dict[ad]
            else:
                dark = rc.get_cal(ad, "processed_dark")

                # If there is no appropriate dark, there is no need to
                # subtract the dark
                if dark is None:
                    log.warning("No changes will be made to %s, since no " \
                                "appropriate dark could be retrieved" \
                                % (ad.filename))
                    # Append the input AstroData object to the list of output
                    # AstroData objects without further processing
                    adoutput_list.append(ad)
                    continue
                else:
                    dark = AstroData(dark)

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
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"], 
                                              strip=True)
            
            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc

    def subtractSky(self, rc):
        """
        This function will subtract the science extension of the input sky
        frames from the science extension of the input science frames. The
        variance and data quality extension will be updated, if they exist.
        
        :param adinput: input science AstroData objects
        :type adinput: Astrodata or Python list of AstroData
        
        :param sky: The input sky frame(s) to be subtracted from the input
                    science frame(s). The input sky frame(s) can be a list of
                    sky filenames, a sky filename, a list of AstroData objects
                    or a single AstroData object. Note: If there are multiple
                    input science frames and one input sky frame provided, then
                    the same sky frame will be applied to all inputs; otherwise
                    the number of input sky frames must match the number of
                    input science frames.
        :type sky: string, Python list of string, AstroData or Python list of
                   Astrodata 
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "subtractSky", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["subtractSky"]
        
        # Initialize the list of output sky corrected AstroData objects
        ad_output_list = []
        
        if rc["sky"]:
            # Use the list of sky frames provided by the user. Generate a
            # dictionary associating the input sky AstroData objects to the
            # input science AstroData objects.
            sky = rc["sky"]
            ad_science_list = rc.get_inputs_as_astrodata()
            for i, ad in enumerate(ad_science_list):
                origname = ad.phu_get_key_value("ORIGNAME")
                if len(sky) == 1:
                    sky_dict.update({origname: sky[0]})
                elif len(sky) == len(ad_science_list):
                    
                    sky_dict.update({origname: RCR.AstroDataRecord(sky[i])})
                else:
                    raise Errors.Error("Number of input sky frames do not "
                                       "match number of input science frames")
        else:
            # The stackSkyFrames primitive puts the dictionary containing the
            # information associating the stacked sky frames to the science
            # frames in the reduction context
            sky_dict = rc["stacked_sky_dict"]
            
        # Loop over each science AstroData object in the science list
        for ad_science in rc.get_inputs_as_astrodata():
            
            # Check whether the subtractSky primitive has been run previously
            timestamp_key = self.timestamp_keys["subtractSky"]
            if ad_science.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has "
                            "already been processed by subtractSky"
                            % (ad_science.filename))
                
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                ad_output_list.append(ad_science)
                continue
            
            # Retrieve the sky AstroData object associated with the input
            # science AstroData object
            origname = ad_science.phu_get_key_value("ORIGNAME")
            if origname in sky_dict:
                ad_sky_for_correction = sky_dict[origname].ad
                
                # Subtract the sky from the input AstroData object
                log.stdinfo("Subtracting the sky (%s) from the science "
                            "AstroData object %s"
                            % (ad_sky_for_correction.filename,
                               ad_science.filename))
                ad_science.sub(ad_sky_for_correction)
            else:
                # There is no appropriate sky for the intput AstroData object
                log.warning("No changes will be made to %s, since no "
                            "appropriate sky could be retrieved"
                            % (ad_science.filename))
                
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                ad_output_list.append(ad_science)
                continue
            
            # Append the output AstroData object to the list of output
            # AstroData objects
            ad_output_list.append(ad_science)
            
        # Add the appropriate time stamp to the PHU and update the filename
        # of the science and sky AstroData objects 
        ad_output_list = gt.finalise_adinput(
          adinput=ad_output_list, timestamp_key=timestamp_key,
          suffix=rc["suffix"])
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(ad_output_list)
        
        yield rc
