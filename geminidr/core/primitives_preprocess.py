import astrodata
import gemini_instruments
from gempy.gemini import gemini_tools as gt
from geminidr.gemini.lookups import DQ_definitions as DQ

import os
import math
import datetime
import numpy as np

from copy import deepcopy

from geminidr import PrimitivesBASE
from .parameters_preprocess import ParametersPreprocess

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class Preprocess(PrimitivesBASE):
    """
    This is the class containing all of the preprocessing primitives
    for the GEMINI level of the type hierarchy tree. It inherits all
    the primitives from the level above, 'GENERALPrimitives'.
    """
    tagset = set(["GEMINI"])

    def __init__(self, adinputs, context, ucals=None, uparms=None):
        super(Preprocess, self).__init__(adinputs, context, ucals=ucals,
                                         uparms=uparms)
        self.parameters = ParametersPreprocess

    
    def ADUToElectrons(self, adinputs=None, stream='main', **params):
        """
        This primitive will convert the units of the pixel data extensions
        of the input AstroData object from ADU to electrons by multiplying
        by the gain.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", "ADUToElectrons", "starting"))
        timestamp_key = self.timestamp_keys["ADUToElectrons"]
        sfx = self.parameters.ADUToElectrons["suffix"]

        for ad in self.adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by ADUToElectrons".
                            format(ad.filename))
                continue

            gain_list = ad.gain()
            # Now multiply the pixel data in each science extension by the gain
            # and the pixel data in each variance extension by the gain squared
            log.status("Converting {} from ADU to electrons by multiplying by "
                       "the gain".format(ad.filename))
            for ext, gain in zip(ad, gain_list):
                extver = ext.hdr.EXTVER
                log.stdinfo("  gain for EXTVER {} = {}".format(extver, gain))
                ext.multiply(gain)
            
            # Update the headers of the AstroData Object. The pixel data now
            # has units of electrons so update the physical units keyword.
            ad.hdr.set('BUNIT', 'electron',
                       keyword_comments=self.keyword_comments)
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(ad, suffix=sfx,  strip=True)

        return
    
    def associateSky(self, adinputs=None, stream='main', **params):
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
        
        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        distance: float
            minimum separation (in arcseconds) required to use an image as sky
        max_skies: int/None
            maximum number of skies to associate to each input frame
        sky: str/list
            name(s) of sky frame(s) to associate to each input
        time: float
            number of seconds
        use_all: bool

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
        log = self.log
        log.debug(gt.log_message("primitive", "associateSky", "starting"))
        timestamp_key = self.timestamp_keys["associateSky"]
        sfx = self.parameters.associateSky["suffix"]
        max_skies = self.parameters.associateSky["max_skies"]
        min_distance = self.parameters.associateSky["distance"]

        # Create a timedelta object using the value of the "time" parameter
        seconds = datetime.timedelta(seconds=self.parameters.associateSky["time"])

        sky = self.parameters.associateSky["sky"]
        if sky:
            # Produce a list of AD objects from the sky frame/list
            ad_sky_list = sky if isinstance(sky, list) else [sky]
            ad_sky_list = [ad if isinstance(ad, astrodata.AstroData) else
                           astrodata.open(ad) for ad in ad_sky_list]
        else:
            #TODO: What replaces this?
            # The separateSky primitive puts the sky AstroData objects in the
            # sky stream. The get_stream function returns a list of AstroData
            # objects when style="AD"
            ad_sky_list = rc.get_stream(stream="sky", style="AD")
        
        if not self.adinputs or not ad_sky_list:
            log.warning("Cannot associate sky frames, since at least one "
                        "science AstroData object and one sky AstroData "
                        "object are required for associateSky")
            
            # Add the science and sky AstroData objects to the output science
            # and sky AstroData object lists, respectively, without further
            # processing, after adding the appropriate time stamp to the PHU
            # and updating the filename.
            if self.adinputs:
                self.adinputs = gt.finalise_adinput(self.adinputs,
                    timestamp_key=timestamp_key, suffix=sfx)
            
            if ad_sky_list:
                ad_sky_output_list = gt.finalise_adinput(
                    adinput=ad_sky_list, timestamp_key=timestamp_key,
                    suffix=sfx)
            else:
                ad_sky_output_list = []
        else:
            # Initialize the dict that will contain the association between
            # the science AstroData objects and the sky AstroData objects
            sky_dict = {}
            
            for ad_sci in self.adinputs:
                # Determine the sky AstroData objects that are associated with
                # this science AstroData object. Initialize the list of sky
                # AstroDataRecord objects
                adr_sky_list = []
                # Since there are no timestamps in these records, keep a list
                # of time offsets in case we need to limit the number of skies
                delta_time_list = []
                
                # Use the ORIGNAME of the science AstroData object as the key
                # of the dictionary 
                origname = ad_sci.phu.get('ORIGNAME')
                
                # If use_all is True, use all of the sky AstroData objects for
                # each science AstroData object
                if self.parameters.associateSky["use_all"]:
                    log.stdinfo("Associating all available sky AstroData "
                                 "objects with {}" .format(ad_sci.filename))

                    #TODO: SORT THIS OUT! It was AstroDataRecords
                    # Set the list of sky AstroDataRecord objects for this
                    # science AstroData object equal to the input list of sky
                    # AstroDataRecord objects
                    #for ad_sky in ad_sky_list:
                    #    adr_sky_list.append(RCR.AstroDataRecord(ad_sky))
                    
                    # Update the dictionary with the list of sky AstroData
                    # objects associated with this science AstroData object
                    sky_dict.update({origname: ad_sky_list})
                else:
                    ad_sci_datetime = ad_sci.ut_datetime()
                    for ad_sky in ad_sky_list:
                        # Make sure the candidate sky exposures actually match
                        # the science configuration (eg. if sequenced over
                        # different filters or exposure times):
                        same_cfg = gt.matching_inst_config(ad_sci, ad_sky,
                            check_exposure=True)

                        # Time difference between science and sky observations
                        ad_sky_datetime = ad_sky.ut_datetime()
                        delta_time = abs(ad_sci_datetime - ad_sky_datetime)
                        
                        # Select only those sky AstroData objects observed
                        # within "time" seconds of the science AstroData object
                        if (same_cfg and delta_time < seconds):
                            
                            # Get the distance between science and sky images
                            delta_x = ad_sci.x_offset() - ad_sky.x_offset()
                            delta_y = ad_sci.y_offset() - ad_sky.y_offset()
                            delta_sky = math.sqrt(delta_x**2 + delta_y**2)
                            if (delta_sky > min_distance):
                                #TODO: ADR issues again
                                adr_sky_list.append(RCR.AstroDataRecord(ad_sky))
                                delta_time_list.append(delta_time)
                                
                    # Now cull the list of associated skies if necessary to
                    # those closest in time to the sceince observation
                    if max_skies is not None and len(adr_sky_list) > max_skies:
                        sorted_list = sorted(zip(delta_time_list, adr_sky_list))
                        adr_sky_list = [x[1] for x in sorted_list[:max_skies]]
                    
                    # Update the dictionary with the list of sky
                    # AstroDataRecord objects associated with this science
                    # AstroData object
                    sky_dict.update({origname: adr_sky_list})
                
                if not sky_dict[origname]:
                    log.warning("No sky frames available for {}".format(origname))
                else:
                    log.stdinfo("The sky frames associated with {} are:".
                                 format(origname))
                    for adr_sky in sky_dict[origname]:
                        log.stdinfo("  {}".format(adr_sky.ad.filename))

            #TODO: Sort this out!
            # Add the appropriate time stamp to the PHU and change the filename
            # of the science and sky AstroData objects 
            self.adinputs = gt.finalise_adinput(self.adinputs,
                                    timestamp_key=timestamp_key, suffix=sfx)
            
            ad_sky_output_list = gt.finalise_adinput(ad_sky_list,
                                    timestamp_key=timestamp_key, suffix=sfx)
            
            # Add the association dictionary to the reduction context
            #rc["sky_dict"] = sky_dict
        
        # Report the list of output sky AstroData objects to the sky stream in
        # the reduction context 
        #rc.report_output(ad_sky_output_list, stream="sky")

        return

    def correctBackgroundToReferenceImage(self, adinputs=None,
                                          stream='main', **params):
        """
        This primitive does an additive correction to a set
        of images to put their sky background at the same level
        as the reference image before stacking.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        remove_zero_level: bool
            if True, set the new background level to zero in all images
            if False, set it to the level of the first image
        """
        log = self.log
        log.debug(gt.log_message("primitive", 
                                 "correctBackgroundToReferenceImage",
                                 "starting"))
        timestamp_key = self.timestamp_keys["correctBackgroundToReferenceImage"]
        sfx = self.parameters.associateSky["suffix"]
        remove_zero_level = self.parameters.associateSky["remove_zero_level"]

        ref_bg = None

        if len(self.adinputs) <= 1:
            log.warning("No correction will be performed, since at least "
                        "two input AstroData objects are required for "
                        "correctBackgroundToReferenceImage")
        # Check that all images have the same number of extensions
        elif not all(len(ad)==len(self.adinputs[0]) for ad in self.adinputs):
            raise IOError("Number of science extensions in input "
                                    "images do not match")
        else:
            # Loop over input files
            ref_bg_list = {}
            for ad in self.adinputs:
                bg_list = gt.measure_bg_from_image(ad, value_only=True)
                # If this is the first (reference) image, set the reference bg levels
                if not ref_bg_list:
                    if remove_zero_level:
                        ref_bg_list = [0] * len(ad)
                    else:
                        ref_bg_list = bg_list.copy()

                for ext, bg, ref in zip(ad, bg_list, ref_bg_list):
                    if bg is None:
                        if 'qa' in self.context:
                            log.warning("Could not get background level from "
                                "{} EXTVER {}".format(ad.filename,
                                                      ext.hdr.EXTVER))
                            continue
                        else:
                            raise LookupError("Could not get background level "
                            "from {} EXTVER {}".format(ad.filename,
                                                       ext.hdr.EXTVER))

                    # Add the appropriate value to this extension
                    log.fullinfo("Background level is {:.0f} for {} EXTVER {}".
                                 format(bg, ad.filename, ext.hdr.EXTVER))
                    difference = ref - bg
                    log.fullinfo("Adding {:.0f} to match reference background "
                                     "level {:.0f}".format(difference, ref))
                    ext.add(difference)
                    ext.hdr.set('SKYLEVEL', ref,
                                comment=self.keyword_comments["SKYLEVEL"])

                # Timestamp the header and update the filename
                gt.mark_history(ad, primname=self.myself(),
                                keyword=timestamp_key)
                ad.filename = gt.filename_updater(ad, suffix=sfx, strip=True)

        return

    def divideByFlat(self, adinputs=None, stream='main', **params):
        """
        This primitive will divide each SCI extension of the inputs by those
        of the corresponding flat. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the division on the data.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        flat: str
            name of flatfield to use
        """
        log = self.log
        log.debug(gt.log_message("primitive", "divideByFlat", "starting"))
        timestamp_key = self.timestamp_keys["divideByFlat"]
        sfx = self.parameters.divideByFlat["suffix"]

        for ad, flat_file in zip(*gt.make_lists(self.adinputs,
                                           self.parameters.divideByFlat["flat"])):
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by divideByFlat".
                            format(ad.filename))
                continue

            if flat_file is None:
                if 'qa' in self.context:
                    log.warning("No changes will be made to {}, since no "
                                "flatfield has been specified".
                                format(ad.filename))
                    continue
                else:
                    raise IOError("No processed flat listed for {}".
                                   format(ad.filename))

            flat = astrodata.open(flat_file)
            # Check the inputs have matching filters, binning, and shapes
            try:
                gt.check_inputs_match(ad, flat)
            except ValueError:
                # Else try to clip the flat frame to the size of the science
                # data (e.g., for GMOS, this allows a full frame flat to
                # be used for a CCD2-only science frame. 
                if 'GSAOI' in ad.tags:
                    flat = gt.clip_auxiliary_data_GSAOI(adinput=ad, 
                                    aux=flat, aux_type="cal",
                                    keyword_comments=self.keyword_comments)
                else:
                    flat = gt.clip_auxiliary_data(adinput=ad, 
                                    aux=flat, aux_type="cal", 
                                    keyword_comments=self.keyword_comments)

                # Check again, but allow it to fail if they still don't match
                gt.check_inputs_match(ad, flat)

            # Do the division
            log.fullinfo("Dividing the input AstroData object {} by this "
                         "flat:\n{}".format(ad.filename, flat.filename))
            ad.divide(flat)

            # Update the header and filename
            ad.phu.set("FLATIM", flat.filename,
                                 comment=self.keyword_comments["FLATIM"])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx,
                                              strip=True)

        return

    def nonlinearityCorrect(self, adinputs=None, stream='main', **params):
        """
        Apply a generic non-linearity correction to data.
        At present (based on GSAOI implementation) this assumes/requires that
        the correction is polynomial. The ad.non_linear_coeffs() descriptor
        should return the coefficients in ascending order of power

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", "nonlinearityCorrect",
                                 "starting"))
        timestamp_key = self.timestamp_keys["nonlinearityCorrect"]
        sfx = self.parameters.nonlinearityCorrect["suffix"]

        for ad in self.adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to %s, since it has "
                            "already been processed by nonlinearityCorrect".
                            format(ad.filename))
                continue
            
            # Get the correction coefficients
            try:
                nonlin_coeffs = ad.nonlinearity_coeffs()
            except:
                log.warning("Unable to obtain nonlinearity coefficients for "
                            "{}".format(ad.filename))
                continue
            
            # It's impossible to do this cleverly with a string of ad.mult()s
            # so use regular maths
            log.status("Applying nonlinearity correction to {}".
                       format(ad.filename))
            for ext, coeffs in zip(ad, nonlin_coeffs):
                log.status("   nonlinearity correction for EXTVER {} is {:s}".
                           format(ext.hdr.EXTVER, coeffs))
                pixel_data = np.zeros_like(ext.data)
                for n in range(len(coeffs), 0, -1):
                    pixel_data += coeffs[n-1]
                    pixel_data *= ext.data
                # Try to do something useful with the VAR plane, if it exists
                # Since the data are fairly pristine, VAR will simply be the
                # Poisson noise (divided by gain if in ADU, divided by COADDS
                # if the coadds are averaged), possibly plus read-noise**2
                # So making an additive correction will sort this out,
                # irrespective of whether there's read noise
                if ext.variance is not None:
                    div_factor = 1
                    bunit  = ext.hdr.get("BUNIT", 'ADU')
                    if bunit.upper() == 'ADU':
                        div_factor *= ext.gain().as_pytype()
                    if not ext.is_coadds_summed().as_pytype():
                        div_factor *= ext.coadds().as_pytype()
                    ext.variance += (pixel_data - ext.data) / div_factor
                # Now update the SCI extension
                ext.data = pixel_data

            # Timestamp the header and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx,
                                              strip=True)
        
        return
    
    def thresholdFlatfield(self, adinputs=None, stream='main', **params):
        """
        This primitive sets the DQ '64' bit for any pixels which have a value
        <lower or >upper in the SCI plane.
        it also sets the science plane pixel value to 1.0 for pixels which are bad
        and very close to zero, to avoid divide by zero issues and inf values
        in the flatfielded science data.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        lower: float
            value below which DQ pixels should be set to unilluminated
        upper: float
            value above which DQ pixels should be set to unilluminated
        """
        log = self.log
        log.debug(gt.log_message("primitive", "thresholdFlatfield",
                                 "starting"))
        timestamp_key = self.timestamp_keys["thresholdFlatfield"]
        sfx = self.parameters.thresholdFlatfield["suffix"]
        lower = self.parameters.thresholdFlatfield["lower"]
        upper = self.parameters.thresholdFlatfield["upper"]

        for ad in self.adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by thresholdFlatfield".
                            format(ad.filename))
                continue

            for ext in ad:
                # Mark the unilumminated pixels with a bit '64' in the DQ plane.
                # make sure the 64 is an int16 64 else it will
                # promote the DQ plane to int64
                unillum = np.where(((ext.data>upper) | (ext.data<lower)) &
                                   (ext.mask | DQ.bad_pixel==0),
                                  np.int16(DQ.unilluminated), np.int16(0))
                ext.mask |= unillum
                log.fullinfo("ThresholdFlatfield set bit '64' for values "
                             "outside the range [{:.2f},{:.2f}]".
                             format(lower, upper))

                # Set the sci value to 1.0 where it is less that 0.001 and
                # where the DQ says it's non-illuminated.
                ext.data[ext.data < 0.001] = 1.0
                ext.data[ext.mask==DQ.unilluminated] = 1.0
                log.fullinfo("ThresholdFlatfield set flatfield pixels to 1.0 "
                             "for values below 0.001 and non-illuminated pixels.")

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"], 
                                              strip=True)

        return

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
                if ad[DQ,extver]:
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
