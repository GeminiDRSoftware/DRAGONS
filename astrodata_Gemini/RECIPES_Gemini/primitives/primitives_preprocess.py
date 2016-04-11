import os
import math
import datetime
import numpy as np

from copy import deepcopy

from astrodata import AstroData
from astrodata.utils import Errors
from astrodata.utils import logutils
from astrodata.utils.gemconstants import SCI, VAR, DQ

from gempy.gemini import gemini_tools as gt

from recipe_system.reduction import reductionContextRecords as RCR

from primitives_GENERAL import GENERALPrimitives
# ------------------------------------------------------------------------------
pkgname =  __file__.split('astrodata_')[1].split('/')[0]

# ------------------------------------------------------------------------------
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
    
    def ADUToElectrons(self, rc):
        """
        This primitive will convert the units of the pixel data extensions
        of the input AstroData object from ADU to electrons by multiplying
        by the gain.
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "ADUToElectrons", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["ADUToElectrons"]
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the ADUToElectrons primitive has been run
            # previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has "
                            "already been processed by ADUToElectrons"
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
            log.status("Converting %s from ADU to electrons by multiplying by "
                       "the gain" % (ad.filename))
            for ext in ad[SCI]:
                extver = ext.extver()
                log.stdinfo("  gain for [%s,%d] = %s" %
                            (SCI, extver, gain.get_value(extver=extver)))
            ad = ad.mult(gain)
            
            # Update the headers of the AstroData Object. The pixel data now
            # has units of electrons so update the physical units keyword.
            gt.update_key(adinput=ad, keyword="BUNIT", value="electron", comment=None,
                          extname=SCI, keyword_comments=self.keyword_comments)
            if ad[VAR]:
                gt.update_key(adinput=ad, keyword="BUNIT", value="electron*electron",
                              comment=None, extname=VAR, 
                              keyword_comments=self.keyword_comments)
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
            
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
            if not isinstance(rc["sky"], list):
                ad_sky_list = [rc["sky"]]
            else:
                ad_sky_list = rc["sky"]
                
            # Convert filenames to AD instances if necessary
            tmp_list = []
            for sky in ad_sky_list:
                if type(sky) is not AstroData:
                    sky = AstroData(sky)
                tmp_list.append(sky)
            ad_sky_list = tmp_list
        else:
            # The separateSky primitive puts the sky AstroData objects in the
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
            else:
                ad_science_output_list = []
            
            if ad_sky_list:
                ad_sky_output_list = gt.finalise_adinput(
                    adinput=ad_sky_list, timestamp_key=timestamp_key,
                    suffix=suffix)
            else:
                ad_sky_output_list = []
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
                    for ad_sky in ad_sky_list:
                        adr_sky_list.append(RCR.AstroDataRecord(ad_sky))
                    
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
                    
                        # Make sure the candidate sky exposures actually match
                        # the science configuration (eg. if sequenced over
                        # different filters or exposure times):
                        same_cfg = gt.matching_inst_config(ad_science, ad_sky,
                            check_exposure=True)

                        # Get the datetime object of the sky AstroData object
                        # using the appropriate descriptor
                        ad_sky_datetime = ad_sky.ut_datetime()
                        
                        # Create a timedelta object using the value of the
                        # "time" parameter
                        seconds = datetime.timedelta(seconds=rc["time"])
                        
                        # Select only those sky AstroData objects observed
                        # within "time" seconds of the science AstroData object
                        if (same_cfg and \
                            abs(ad_science_datetime - ad_sky_datetime) \
                            < seconds):
                            
                            # Get the distance of the science and sky AstroData
                            # objects using the x_offset and y_offset
                            # descriptors
                            delta_x = ad_science.x_offset() - ad_sky.x_offset()
                            delta_y = ad_science.y_offset() - ad_sky.y_offset()
                            delta_sky = math.sqrt(delta_x**2 + delta_y**2)
                            
                            # Select only those sky AstroData objects that are
                            # greater than "distance" arcsec away from the
                            # science AstroData object
                            if (delta_sky > rc["distance"]):
                                adr_sky_list.append(RCR.AstroDataRecord(ad_sky))
                    
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
        next = np.array([ad.count_exts(SCI) for ad in adinput])

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

            # Check if the images have been sky subtracted. If so, 
            # check whether detectSources has been run more recently. 
            # If not, it must be run to ensure that measureBG is working 
            # on the most recent catalog. 
            # NOTE: This check should really be replaced with a call to
            # a new general background calculating primitive that simply
            # takes the median of the background with the stars masked
            # (the recentness of the catalog shouldn't matter too much 
            # for this).
            rerun_ds = False
            for ad in adinput:
                subsky = ad.phu_get_key_value('SUBSKY')
                if subsky is not None:
                    detecsrc = ad.phu_get_key_value('DETECSRC')
                    if detecsrc is not None:
                        if subsky > detecsrc:
                            rerun_ds = True
                            break
                    else:
                        rerun_ds = True
                        break
            if rerun_ds:
                log.fullinfo("This data has been sky subtracted, so the " \
                             "background level will be re-measured") 
                rc.run("detectSources")
                rc.run("measureBG(separate_ext=True,remove_bias=False)")
            else:
                # Check whether measureBG needs to be run
                bg_list = [sciext.get_key_value("SKYLEVEL") \
                           for ad in adinput for sciext in ad[SCI]]
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
                for sciext in ad[SCI]:
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
                gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
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
                if 'GSAOI' in ad.types:
                    flat = gt.clip_auxiliary_data_GSAOI(adinput=ad, 
                                    aux=flat, aux_type="cal",
                                    keyword_comments=self.keyword_comments)[0]
                else:
                    flat = gt.clip_auxiliary_data(adinput=ad, 
                                    aux=flat, aux_type="cal", 
                                    keyword_comments=self.keyword_comments)[0]

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
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

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
        Apply a generic non-linearity correction to data.
        At present (based on GSAOI implementation) this assumes/requires that
        the correction is polynomial. The ad.non_linear_coeffs() descriptor
        should return the coefficients in ascending order of power
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
            
            if ad['VAR'] is not None:
                log.warning("%s has a VAR extension, which will be rendered "
                            "meaningless by nonlinearityCorrect"
                            % (ad.filename))

            # Get the coefficients from the lookup table
            nonlin_coeffs = ad.nonlinearity_coeffs()
            
            # It's impossible to do this cleverly with a string of ad.mult()s
            # so use numpy. That's OK because if there's already a VAR here
            # something's gone wrong, so only SCI will has to be altered
            log.status("Applying nonlinearity correction to %s "
                       % (ad.filename))
            for ext in ad[SCI]:
                extver = ext.extver()
                coeffs = nonlin_coeffs.get_value(extver=extver)
                log.status("   nonlinearity correction for [%s,%d] is %s" %
                           (SCI, extver, coeffs))
                pixel_data = np.zeros_like(ext.data)
                for n in range(len(coeffs),0,-1):
                    pixel_data += coeffs[n-1]
                    pixel_data *= ext.data
                ext.data = pixel_data

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

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
    
    def thresholdFlatfield(self, rc):
        """
        This primitive sets the DQ '64' bit for any pixels which have a value
        <lower or >upper in the SCI plane.
        it also sets the science plane pixel value to 1.0 for pixels which are bad
        and very close to zero, to avoid divide by zero issues and inf values
        in the flatfielded science data.
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "thresholdFlatfield", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["thresholdFlatfield"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the normalizeFlat primitive has been run previously
            # ASK EMMA if we need this message
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by thresholdFlatfield" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Loop over each science extension in each input AstroData object
            upper = rc['upper']
            lower = rc['lower']
            for ext in ad[SCI]:
                
                extver = ext.extver()
                sci_data = ext.data
                dq_data = ad[DQ,extver].data

                # Mark the unilumminated pixels with a bit '64' in the DQ plane.
                # make sure the 64 is an int16 64 else it will promote the DQ plane to int64
                unilum = np.where(((sci_data>upper) | (sci_data<lower)) &
                                   (dq_data | 1==0), np.int16(64), np.int16(0))

                dq_data = np.bitwise_or(dq_data,unilum)

                # Now replace the DQ data
                ad[DQ,extver].data = dq_data

                log.fullinfo("ThresholdFlatfield set bit '64' for values"
                             " outside the range [%.2f,%.2f]"%(lower,upper))

                # Set the sci value to 1.0 where it is less that 0.001 and
                # where the DQ says it's non-illuminated.
                sci_data = np.where(sci_data < 0.001, 1.0, sci_data)
                sci_data = np.where(dq_data == 64, 1.0, sci_data)
                ad['SCI',extver].data=sci_data

                log.fullinfo("ThresholdFlatfield set flatfield pixels to 1.0"
                             " for values below 0.001 and non-illuminated pixels.")


            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

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
            for ext in ad[SCI]:
                extver = ext.extver()
                
                # Normalise the input AstroData object. Calculate the median
                # value of the science extension
                if ad[DQ,extver] is not None:
                    median = np.median(ext.data[np.where(ad[DQ,extver].data == 0)]
                                       ).astype(np.float64)
                else:
                    median = np.median(ext.data).astype(np.float64)
                # Divide the science extension by the median value of the science
                # extension, and the VAR (if it exists) by the square of this
                log.fullinfo("Normalizing %s[%s,%d] by dividing by the median "
                             "= %f" % (ad.filename, ext.extname(),
                                       extver, median))
                ext.data /= median
                # Take care of the VAR as well!
                if ad[VAR,extver] is not None:
                    ad[VAR,extver].data /= median*median

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

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

#    def scaleByExposureTime(self, rc):
#        """
#        This primitive scales input images to match the exposure time of
#        the first image. 
#        """
#        # Instantiate the log
#        log = logutils.get_logger(__name__)
#
#        # Log the standard "starting primitive" debug message
#        log.debug(gt.log_message("primitive", "scaleByExposureTime", "starting"))
#
#        # Define the keyword to be used for the time stamp for this primitive
#        timestamp_key = self.timestamp_keys["scaleByExposureTime"]
#
#        # Initialize the list of output AstroData objects
#        adoutput_list = []
#
#        # First check if any scaling is actually required
#        exptimes = []
#        for ad in rc.get_inputs_as_astrodata():
#            exptimes.append(ad.exposure_time())
#        if set(exptimes) == 1:
#            log.fullinfo("Exposure times are the same therefore no scaling"
#                         "is required.")
#            adoutput_list = rc.get_inputs_as_astrodata()
#        else:
#            first = True
#            reference_exptime = 1.0
#            # Loop over each input AstroData object in the input list
#            for ad in rc.get_inputs_as_astrodata():
#
#                exptime = ad.exposure_time()
#                # Scale by the relative exposure time
#                if first:
#                    reference_exptime = exptime
#                    scale = 1.0
#                    first = False
#                    first_filename = ad.filename
#                else:
#                    scale = reference_exptime / exptime
#
#                # Log and save the scale factor. Also change the exposure time
#                # (not sure if this is OK, since I'd rather leave this as the
#                # original value, but a lot of primitives match/select on 
#                # this - ED)
#                log.fullinfo("Intensity scaled to match exposure time of %s: "
#                             "%.3f" % (first_filename, scale))
##                ad.phu_set_key_value("EXPSCALE", scale,
##                                 comment=self.keyword_comments["EXPSCALE"])
#                ad.phu_set_key_value("EXPTIME", reference_exptime.as_pytype(),
#                                 comment=self.keyword_comments["EXPTIME"])
#
#                # Multiply by the scaling factor
#                ad.mult(scale)
#
#                # Add the appropriate time stamps to the PHU
#                gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
#
#                # Change the filename
#                ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"], 
#                                              strip=True)
#
#                # Append the output AstroData object to the list
#                # of output AstroData objects
#                adoutput_list.append(ad)
#
#        # Report the list of output AstroData objects to the reduction
#        # context
#        rc.report_output(adoutput_list)
#        
#        yield rc     


    def separateSky(self, rc):

        """
        Given a set of input exposures, sort them into separate but
        possibly-overlapping streams of on-target and sky frames. This is
        achieved by dividing the data into distinct pointing/dither groups,
        applying a set of rules to classify each group as target(s) or sky
        and optionally overriding those classifications with user guidance
        (up to and including full manual specification of both lists).

        If all exposures are found to be on source then both output streams
        will replicate the input. Where a dataset appears in both lists, a
        separate copy (TBC: copy-on-write?) is made in the sky list to avoid
        subsequent operations on one of the output lists affecting the other.

        The following optional parameters are accepted, in addition to those
        common to other primitives:

        :param frac_FOV: Proportion by which to scale the instrumental field
            of view when determining whether points are considered to be
            within the same field, for tweaking borderline cases (eg. to avoid
            co-adding target positions right at the edge of the field).
        :type frac_FOV: float

        :param ref_obj: Exposure filenames (as read from disk, without any
            additional suffixes appended) to be considered object/on-target
            exposures, as overriding guidance for any automatic classification.
        :type ref_obj: string with comma-separated names

        :param ref_sky: Exposure filenames to be considered sky exposures, as
            overriding guidance for any automatic classification.
        :type ref_obj: string with comma-separated names

        :returns: Separate object and sky streams containing AstroData objects

        Any existing OBJFRAME or SKYFRAME flags in the input meta-data will
        also be respected as input (unless overridden by ref_obj/ref_sky) and
        these same keywords are set in the output, along with a group number
        with which each exposure is associated (EXPGROUP).
        """

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

        # Allow tweaking what size of offset, as a fraction of the field
        # dimensions, is considered to move a target out of the field in
        # gt.group_exposures(). If we want to check this parameter value up
        # front I'm assuming the infrastructure will do that at some point.
        frac_FOV = rc["frac_FOV"]

        # Get optional user-specified lists of object or sky filenames, to
        # assist with classifying groups as one or the other. Currently
        # primitive parameters have no concept of lists so we parse a string
        # argument here. As of March 2014 this only works with "reduce2":
        ref_obj = rc["ref_obj"].split(',')
        ref_sky = rc["ref_sky"].split(',')
        if ref_obj == ['']: ref_obj = []
        if ref_sky == ['']: ref_sky = []

        # Loop over input AstroData objects and extract their filenames, for
        # 2-way comparison with any ref_obj/ref_sky list(s) supplied by the
        # user. This could go into a support function but should become fairly
        # trivial once we address the issues noted below.
        filenames = {}
        for ad in ad_input_list:

            # For the time being we directly look up the filename provided
            # for this exposure (without whatever pipeline suffixes happen
            # to have been added during prior processing steps) in order to
            # match the ref_obj & ref_sky parameters, due to some bug in
            # what the AstroData API is returning (related to Trac #446):
            base_name = ad.phu_get_key_value("ORIGNAME")

            # Strip any .fits extension from the name we want to refer to.
            # Having this buried in the code should be harmless even if we
            # support other file types later, as I'm told the intention is
            # to remove the file extension from ORIGNAME anyway, which will
            # turn this into a no-op. That change also means we couldn't
            # safely strip .* here anyway. Once this ORIGNAME change
            # happens, the following 2 lines can be removed:
            if base_name.endswith(".fits"):
                base_name = base_name[:-5]

            filenames[ad] = base_name

        # Warn the user if they referred to non-existent input file(s):
        missing = [name for name in ref_obj if name not in filenames.values()]
        missing.extend([name for name in ref_sky \
                       if name not in filenames.values()])

        if missing:
            log.warning("Failed to find the following file(s), specified "\
                "via ref_obj/ref_sky parameters, in the input:")

            for name in missing:
                log.warning("  %s" % name)

        # Loop over input AstroData objects and apply any overriding sky/object
        # classifications based on user-supplied or guiding information:
        for ad in ad_input_list:            

            # Get the corresponding filename, determined above:
            base_name = filenames[ad]

            # Remember any pre-existing classifications so we can point them
            # out if the user requests something different this time:
            if ad.phu_get_key_value("OBJFRAME"):
                obj = True
            else:
                obj = False
            if ad.phu_get_key_value("SKYFRAME"):
                sky = True
            else:
                sky = False

            # If the user specified manually that this file is object and/or
            # sky, note that in the metadata (alongside any existing flags):
            if base_name in ref_obj and not obj:
                if sky:
                    log.warning("%s previously classified as SKY; added" \
                      " OBJECT as requested" % base_name)
                ad.phu_set_key_value("OBJFRAME", "TRUE")

            if base_name in ref_sky and not sky:
                if obj:
                    log.warning("%s previously classified as OBJECT; added" \
                      " SKY as requested" % base_name)
                ad.phu_set_key_value("SKYFRAME", "TRUE")

            # If the exposure is unguided, classify it as sky unless the
            # user has specified otherwise (in which case we loudly point
            # out the anomaly but do as requested).
            #
            # Although this descriptor naming may not always be ideal for
            # non-Gemini applications, it is our AstroData convention for
            # determining whether an exposure is guided or not.
            if ad.wavefront_sensor().is_none():

                # Old Gemini data are missing the guiding keywords but
                # the descriptor then returns None, which is indistinguishable
                # from an unguided exposure (Trac #416). Even phu_get_key_value
                # fails to make this distinction. Whatever we do here (eg.
                # using dates to figure out whether the keywords should be
                # present), it is bound to be Gemini-specific until the
                # behaviour of descriptors is fixed.
                if 'PWFS1_ST' in ad.phu.header and \
                   'PWFS2_ST' in ad.phu.header and \
                   'OIWFS_ST' in ad.phu.header:

                    # The exposure was really unguided:
                    if ad.phu_get_key_value("OBJFRAME"):
                        log.warning("Exp. %s manually flagged as " \
                            "on-target but unguided!" % base_name)
                    else:
                        log.fullinfo("Treating %s as sky since it's unguided" \
                          % base_name)
                        ad.phu_set_key_value("SKYFRAME", "TRUE")

                # (else can't determine guiding state reliably so ignore it)

        # Analyze the spatial clustering of exposures and attempt to sort them
        # into dither groups around common nod positions.
        groups = gt.group_exposures(ad_input_list, pkgname, frac_FOV=frac_FOV)
        ngroups = len(groups)

        log.fullinfo("Identified %d group(s) of exposures" % ngroups)

        # Loop over the nod groups identified above, record which group each
        # exposure belongs to, propagate any already-known classification(s)
        # to other members of the same group and determine whether everything
        # is finally on source and/or sky:
        haveobj = False; havesky = False
        allobj = True; allsky = True
        for group, num in zip(groups, range(ngroups)):
            adlist = group.list()
            obj = False; sky = False
            for ad in adlist:
                ad.phu_set_key_value("EXPGROUP", num)
                if ad.phu_get_key_value("OBJFRAME"): obj = True
                if ad.phu_get_key_value("SKYFRAME"): sky = True
                # if obj and sky: break  # no: need to record all group nums
            if obj:
                haveobj = True
                for ad in adlist:
                    ad.phu_set_key_value("OBJFRAME", "TRUE")
            else:
                allobj = False
            if sky:
                havesky = True
                for ad in adlist:
                    ad.phu_set_key_value("SKYFRAME", "TRUE")
            else:
                allsky = False

        # If we now have object classifications but no sky, or vice versa,
        # make whatever reasonable inferences we can about the others:
        if haveobj and not havesky:
            for ad in ad_input_list:
                if allobj or not ad.phu_get_key_value("OBJFRAME"):
                    ad.phu_set_key_value("SKYFRAME", "TRUE")
        elif havesky and not haveobj:
            for ad in ad_input_list:
                if allsky or not ad.phu_get_key_value("SKYFRAME"):
                    ad.phu_set_key_value("OBJFRAME", "TRUE")

        # If all the exposures are still unclassified at this point, we
        # couldn't decide which groups are which based on user input or guiding
        # so use the distance from the target or failing that assume everything
        # is on source but warn the user about it if there's more than 1 group:
        if not haveobj and not havesky:

            ngroups = len(groups)

            # With 2 groups, the one closer to the target position must be
            # on source and the other is presumably sky. For Gemini data, the
            # former, on-source group should be the one with smaller P/Q.
            # TO DO: Once we update ExposureGroup to use RA & Dec descriptors
            # instead of P & Q, this will need changing to subtract the target
            # RA & Dec explicitly. For non-Gemini data where the target RA/Dec
            # are unknown, we'll have to skip this bit and proceed to assuming
            # everything is on source unless given better information.
            if ngroups == 2:

                log.fullinfo("Treating 1 group as object & 1 as sky, based " \
                  "on target proximity")

                dsq0 = sum([x**2 for x in groups[0].group_cen])
                dsq1 = sum([x**2 for x in groups[1].group_cen])
                if dsq1 < dsq0:
                    order = ["SKYFRAME", "OBJFRAME"]
                else:
                    order = ["OBJFRAME", "SKYFRAME"]

                for group, key in zip(groups, order):
                    adlist = group.list()
                    for ad in adlist:
                        ad.phu_set_key_value(key, "TRUE")

            # For more or fewer than 2 groups, we just have to assume that
            # everything is on target, for lack of better information. With
            # only 1 group, this should be a sound assumption, otherwise we
            # warn the user.
            else:
                if ngroups > 1:
                    log.warning("Unable to determine which of %d detected " \
                        "groups are sky/object -- assuming they are all on " \
                        "target AND usable as sky" % ngroups)
                else:
                    log.fullinfo("Treating a single group as both object & sky")

                for ad in ad_input_list:
                    ad.phu_set_key_value("OBJFRAME", "TRUE")
                    ad.phu_set_key_value("SKYFRAME", "TRUE")

        # It's still possible for some exposures to be unclassified at this
        # point if the user has identified some but not all of several groups
        # manually (or that's what's in the headers). We can't do anything
        # sensible to rectify that, so just discard the unclassified ones and
        # complain about it.
        missing = []
        for ad in ad_input_list:
            if not ad.phu_get_key_value("OBJFRAME") and \
               not ad.phu_get_key_value("SKYFRAME"):
                missing.append(filenames[ad])
        if missing:
            log.warning("ignoring the following input file(s), which could " \
              "not be classified as object or sky after applying incomplete " \
              "prior classifications from the input:")
            for name in missing:
                log.warning("  %s" % name)

        # Construct object & sky lists from the classifications stored above
        # in exposure meta-data, making a complete copy of the input for any
        # duplicate entries (it is hoped this won't require additional memory
        # once memory mapping is used appropriately):
        ad_science_list = []
        ad_sky_list = []
        for ad in ad_input_list:
            on_source = ad.phu_get_key_value("OBJFRAME")
            if on_source:
                ad_science_list.append(ad)
            if ad.phu_get_key_value("SKYFRAME"):
                if on_source:
                    ad_sky_list.append(deepcopy(ad))
                else:
                    ad_sky_list.append(ad)

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
          suffix=suffix, allow_empty=True)
            
        ad_sky_output_list = gt.finalise_adinput(
          adinput=ad_sky_list, timestamp_key=timestamp_key, suffix=suffix,
          allow_empty=True)
               
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
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

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
        sky_dict = {}
        
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


    def subtractSkyBackground(self, rc):
        """
        This primitive is used to subtract the sky background specified by 
        the keyword SKYLEVEL.
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "subtractSkyBackground", "starting"))

        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["subtractSkyBackground"]

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            # Check whether the myScienceStep primitive has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has "
                            "already been processed by subtractSkyBackground"
                            % ad.filename)

                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Get the sky background
            for sciext in ad[SCI]:
                bg = sciext.get_key_value("SKYLEVEL")
                
                if bg is None:
                    log.warning("No changes will be made to %s, since there "
                                "is no sky background measured" % ad.filename)
                else:    
                    log.fullinfo("Subtracting %.0f to remove sky level from "
                                 "image %s" % (bg, ad.filename))
                    sciext.sub(bg)
                    
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)

            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)

        # Report the list of output AstroData objects to the reduction context
        rc.report_output(adoutput_list)
        
        yield rc
