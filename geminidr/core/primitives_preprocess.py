#
#                                                                  gemini_python
#
#                                                       primitives_preprocess.py
# ------------------------------------------------------------------------------
import math
import datetime
import numpy as np
from copy import deepcopy

import astrodata
import gemini_instruments

from gempy.gemini import gemini_tools as gt
from geminidr.gemini.lookups import DQ_definitions as DQ

from geminidr import PrimitivesBASE
from .parameters_preprocess import ParametersPreprocess

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class Preprocess(PrimitivesBASE):
    """
    This is the class containing all of the preprocessing primitives.
    """
    tagset = None

    def __init__(self, adinputs, **kwargs):
        super(Preprocess, self).__init__(adinputs, **kwargs)
        self.parameters = ParametersPreprocess

    def addObjectMaskToDQ(self, adinputs=None, **params):
        """
        This primitive combines the object mask in a OBJMASK extension
        into the DQ plane.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        sfx = params["suffix"]
        for ad in adinputs:
            for ext in ad:
                if hasattr(ext, 'OBJMASK'):
                    if ext.mask is None:
                        ext.mask = deepcopy(ext.OBJMASK)
                    else:
                        # CJS: This probably shouldn't just be dumped into
                        # the 1-bit
                        ext.mask |= ext.OBJMASK
                else:
                    log.warning('No object mask present for {}:{}; cannot '
                                'apply object mask'.format(ad.filename,
                                                           ext.hdr['EXTVER']))
            ad.filename = gt.filename_updater(ad, suffix=sfx, strip=True)
        return adinputs

    def ADUToElectrons(self, adinputs=None, **params):
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
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]

        for ad in adinputs:
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
                extver = ext.hdr['EXTVER']
                log.stdinfo("  gain for EXTVER {} = {}".format(extver, gain))
                ext.multiply(gain)
            
            # Update the headers of the AstroData Object. The pixel data now
            # has units of electrons so update the physical units keyword.
            ad.hdr.set('BUNIT', 'electron', self.keyword_comments['BUNIT'])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(ad, suffix=sfx,  strip=True)
        return adinputs
    
    def applyDQPlane(self, adinputs=None, **params):
        """
        This primitive sets the value of pixels in the science plane according
        to flags from the DQ plane.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        replace_flags: int
            The DQ bits, of which one needs to be set for a pixel to be replaced
        replace_value: str/float
            "median" or "average" to replace with that value of the good pixels,
            or a value
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        replace_flags = params["replace_flags"]
        replace_value = params["replace_value"]

        flag_list = [int(math.pow(2,i)) for i,digit in
                enumerate(str(bin(replace_flags))[2:][::-1]) if digit=='1']
        log.stdinfo("The flags {} will be applied".format(flag_list))

        for ad in adinputs:
            for ext in ad:
                if ext.mask is None:
                    log.warning("No DQ plane exists for {}:{}, so the correction "
                                "cannot be applied".format(ad.filename,
                                                           ext.hdr['EXTVER']))
                    continue

                if replace_value in ['median', 'average']:
                    oper = getattr(np, replace_value)
                    rep_value = oper(ext.data[ext.mask & replace_flags == 0])
                    log.fullinfo("Replacing bad pixels in {}:{} with the {} "
                                 "of the good data".format(ad.filename,
                                            ext.hdr['EXTVER'], replace_value))
                else:
                    try:
                        rep_value = float(replace_value)
                        log.fullinfo("Replacing bad pixels in {}:{} with the "
                                     "user value {}".format(ad.filename,
                                           ext.hdr['EXTVER'], rep_value))
                    except:
                        log.warning("Value for replacement should be 'median', "
                                    "'average', or a number")
                        continue

                ext.data[ext.mask & replace_flags != 0] = rep_value

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(ad, suffix=params["suffix"], strip=True)
        return adinputs

    def associateSky(self, adinputs=None, **params):
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
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        max_skies = params["max_skies"]
        min_distance = params["distance"]

        # Create a timedelta object using the value of the "time" parameter
        seconds = datetime.timedelta(seconds=params["time"])

        if 'sky' in params and params['sky']:
            sky = params['sky']
            # Produce a list of AD objects from the sky frame/list
            ad_sky_list = sky if isinstance(sky, list) else [sky]
            ad_sky_list = [ad if isinstance(ad, astrodata.AstroData) else
                           astrodata.open(ad) for ad in ad_sky_list]
        else:
            # The separateSky primitive puts the sky AstroData objects in the
            # sky stream.
            ad_sky_list = self.streams.get('sky')
        
        if not adinputs or not ad_sky_list:
            log.warning("Cannot associate sky frames, since at least one "
                        "science AstroData object and one sky AstroData "
                        "object are required for associateSky")
            
            # Add the science and sky AstroData objects to the output science
            # and sky AstroData object lists, respectively, without further
            # processing, after adding the appropriate time stamp to the PHU
            # and updating the filename.
            if adinputs:
                adinputs = gt.finalise_adinput(adinputs,
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
            
            for ad_sci in adinputs:
                # Determine the sky AstroData objects that are associated with
                # this science AstroData object. Initialize the list of sky
                # AstroDataRecord objects
                this_sky_list = []
                # Since there are no timestamps in these records, keep a list
                # of time offsets in case we need to limit the number of skies
                delta_time_list = []
                
                # Use the ORIGNAME of the science AstroData object as the key
                # of the dictionary 
                origname = ad_sci.phu.get('ORIGNAME')
                
                # If use_all is True, use all of the sky AstroData objects for
                # each science AstroData object
                if params["use_all"]:
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
                                this_sky_list.append(ad_sky)
                                delta_time_list.append(delta_time)
                                
                    # Now cull the list of associated skies if necessary to
                    # those closest in time to the sceince observation
                    if max_skies is not None and len(this_sky_list) > max_skies:
                        sorted_list = sorted(zip(delta_time_list, this_sky_list))
                        this_sky_list = [x[1] for x in sorted_list[:max_skies]]

                    # Update the dictionary with the list of sky
                    # AstroDataRecord objects associated with this science
                    # AstroData object
                    sky_dict.update({origname: this_sky_list})
                
                if not sky_dict[origname]:
                    log.warning("No sky frames available for {}".format(origname))
                else:
                    log.stdinfo("The sky frames associated with {} are:".
                                 format(origname))
                    for ad_sky in sky_dict[origname]:
                        log.stdinfo("  {}".format(ad_sky.filename))

            #TODO: Sort this out!
            # Add the appropriate time stamp to the PHU and change the filename
            # of the science and sky AstroData objects 
            adinputs = gt.finalise_adinput(adinputs,
                                    timestamp_key=timestamp_key, suffix=sfx)
            ad_sky_output_list = gt.finalise_adinput(ad_sky_list,
                                    timestamp_key=timestamp_key, suffix=sfx)
            
            # Store the association dictionary as an attribute of the class
            self.sky_dict = sky_dict
        
        # Report the list of output sky AstroData objects to the sky stream in
        # the reduction context 
        self.streams['sky'] = ad_sky_output_list
        return adinputs

    def correctBackgroundToReferenceImage(self, adinputs=None, **params):
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
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        remove_zero_level = params["remove_zero_level"]

        if len(adinputs) <= 1:
            log.warning("No correction will be performed, since at least "
                        "two input AstroData objects are required for "
                        "correctBackgroundToReferenceImage")
        # Check that all images have the same number of extensions
        elif not all(len(ad)==len(adinputs[0]) for ad in adinputs):
            raise IOError("Number of science extensions in input "
                                    "images do not match")
        else:
            # Loop over input files
            ref_bg_list = []
            for ad in adinputs:
                bg_list = gt.measure_bg_from_image(ad, value_only=True)
                # If this is the first (reference) image, set the reference bg levels
                if not ref_bg_list:
                    if remove_zero_level:
                        ref_bg_list = [0] * len(ad)
                    else:
                        ref_bg_list = bg_list

                for ext, bg, ref in zip(ad, bg_list, ref_bg_list):
                    if bg is None:
                        if 'qa' in self.context:
                            log.warning("Could not get background level from "
                                "{}:{}".format(ad.filename, ext.hdr['EXTVER']))
                            continue
                        else:
                            raise LookupError("Could not get background level "
                            "from {}:{}".format(ad.filename, ext.hdr['EXTVER']))

                    # Add the appropriate value to this extension
                    log.fullinfo("Background level is {:.0f} for {}:{}".
                                 format(bg, ad.filename, ext.hdr['EXTVER']))
                    difference = np.float32(ref - bg)
                    log.fullinfo("Adding {:.0f} to match reference background "
                                     "level {:.0f}".format(difference, ref))
                    ext.add(difference)
                    ext.hdr.set('SKYLEVEL', ref,
                                self.keyword_comments["SKYLEVEL"])

                # Timestamp the header and update the filename
                gt.mark_history(ad, primname=self.myself(),
                                keyword=timestamp_key)
                ad.filename = gt.filename_updater(ad, suffix=sfx, strip=True)

        return adinputs

    def darkCorrect(self, adinputs=None, **params):
        """
        Obtains processed dark(s) from the calibration service and subtracts
        it/them from the science image(s).

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        dark: str/list
            name of dark to use (in which case the cal request is superfluous)
        """
        self.getProcessedDark(adinputs)
        adinputs = self.subtractDark(adinputs, **params)
        return adinputs

    def divideByFlat(self, adinputs=None, **params):
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
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        flat_list = params["flat"] if params["flat"] else [
            self._get_cal(ad, 'processed_flat') for ad in adinputs]

        # Provide a flatfield AD object for every science frame
        for ad, flat in zip(*gt.make_lists(adinputs, flat_list,
                                           force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by divideByFlat".
                            format(ad.filename))
                continue

            if flat is None:
                if 'qa' in self.context:
                    log.warning("No changes will be made to {}, since no "
                                "flatfield has been specified".
                                format(ad.filename))
                    continue
                else:
                    raise IOError("No processed flat listed for {}".
                                   format(ad.filename))

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
            ad.phu.set("FLATIM", flat.filename, self.keyword_comments["FLATIM"])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=params["suffix"],
                                              strip=True)
        return adinputs

    def flatCorrect(self, adinputs=None, **params):
        """
        Obtains processed flat(s) from the calibration service and divides
        the science image(s) by it/them.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        flat: str/list
            name of flat to use (in which case the cal request is superfluous)
        """
        self.getProcessedFlat(adinputs)
        adinputs = self.divideByFlat(adinputs, **params)
        return adinputs

    def makeSky(self, adinputs=None, **params):
        adinputs = self.separateSky(adinputs, **params)
        adinputs = self.associateSky(adinputs, **params)
        adinputs = self.stackSkyFrames(adinputs, **params)
        #self.makeMaskedSky()
        return adinputs

    def nonlinearityCorrect(self, adinputs=None, **params):
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
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]

        for ad in adinputs:
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
                           format(ext.hdr['EXTVER'], coeffs))
                pixel_data = np.zeros_like(ext.data)

                # Convert back to ADU per exposure if coadds have been summed
                # or if the data have been converted to electrons
                bunit = ext.hdr.get("BUNIT", 'ADU').upper()
                conv_factor = 1 if bunit == 'ADU' else ext.gain()
                if ext.is_coadds_summed():
                    conv_factor *= ext.coadds()
                for n in range(len(coeffs), 0, -1):
                    pixel_data += coeffs[n-1]
                    pixel_data *= ext.data / conv_factor
                pixel_data *= conv_factor
                # Try to do something useful with the VAR plane, if it exists
                # Since the data are fairly pristine, VAR will simply be the
                # Poisson noise (divided by gain if in ADU, divided by COADDS
                # if the coadds are averaged), possibly plus read-noise**2
                # So making an additive correction will sort this out,
                # irrespective of whether there's read noise
                conv_factor = ext.gain() if bunit == 'ADU' else 1
                if not ext.is_coadds_summed():
                    conv_factor *= ext.coadds()
                if ext.variance is not None:
                    ext.variance += (pixel_data - ext.data) / conv_factor
                # Now update the SCI extension
                ext.data = pixel_data

            # Timestamp the header and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx,
                                              strip=True)
        return adinputs
    
    def normalizeFlat(self, adinputs=None, **params):
        """
        This primitive normalizes each science extension of the input
        AstroData object by its mean

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        scale: str
            type of scaling to use. Must be a numpy function
        separate_ext: bool
            Scale each extension individually?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        separate_ext = params["separate_ext"]
        operator = getattr(np, params["scale"], None)
        if not callable(operator):
            log.warning("Operator {} not found, defaulting to median".
                        format(params["scale"]))
            operator = np.median

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by normalizeFlat".
                            format(ad.filename))
                continue

            if separate_ext:
                for ext in ad:
                    # Normalise the input AstroData object. Calculate the
                    # "average" value of the science extension
                    if ext.mask is None:
                        scaling = operator(ext.data).astype(np.float32)
                    else:
                        scaling = operator(ext.data[ext.mask==0]).astype(np.float32)
                    # Divide the science extension by the median value
                    # VAR is taken care of automatically
                    log.fullinfo("Normalizing {} EXTVER {} by dividing by {:.2f}".
                                 format(ad.filename, ext.hdr['EXTVER'], scaling))
                    ext /= scaling
            else:
                # Combine pixels from all extensions, using DQ if present
                scaling = operator(np.concatenate([(ext.data.ravel()
                        if ext.mask is None else ext.data[ext.mask==0].ravel())
                                            for ext in ad])).astype(np.float32)
                log.fullinfo("Normalizing {} by dividing by {:.2f}".
                            format(ad.filename, scaling))
                ad /= scaling

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx,
                                              strip=True)
        return adinputs

#### Refactored but not tested
#    def scaleByExposureTime(self, adinputs=None, **params):
#        """
#        This primitive scales input images to match the exposure time of
#        the first image. 
#        """
#        log = self.log
#        log.debug(gt.log_message("primitive", "scaleByExposureTime", "starting"))
#        timestamp_key = self.timestamp_keys["scaleByExposureTime"]
#        sfx = params["suffix"]
#
#        # First check if any scaling is actually required
#        exptimes = [ad.exposure_time() for ad in adinputs]
#        if len(set(exptimes)) == 1:
#            log.fullinfo("Exposure times are the same therefore no scaling "
#                         "is required.")
#        else:
#            reference_exptime = None
#            # Loop over each input AstroData object in the input list
#            for ad in sadinputs:
#                exptime = ad.exposure_time()
#                # Scale by the relative exposure time
#                if reference_exptime is None:
#                    reference_exptime = exptime
#                    scale = 1.0
#                    first_filename = ad.filename
#                else:
#                    scale = reference_exptime / exptime
#
#                # Log and save the scale factor. Also change the exposure time
#                # (not sure if this is OK, since I'd rather leave this as the
#                # original value, but a lot of primitives match/select on 
#                # this - ED)
#                log.fullinfo("Intensity scaled to match exposure time of {}: "
#                             "{:.3f}".format(first_filename, scale))
##                ad.phu.set("EXPSCALE", scale,
##                                 comment=self.keyword_comments["EXPSCALE"])
#                ad.phu.set("EXPTIME", reference_exptime,
#                           comment=self.keyword_comments["EXPTIME"])
#
#                # Multiply by the scaling factor
#                ad.mult(scale)
#
#                # Timestamp and update the filename
#                gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
#                ad.filename = gt.filename_updater(adinput=ad, suffix=sfx,
#                                              strip=True)
#        return adinputs

    def separateSky(self, adinputs=None, **params):
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

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        frac_FOV: float
            Proportion by which to scale the instrumental field of view when
            determining whether points are considered to be within the same
            field, for tweaking borderline cases (eg. to avoid co-adding
            target positions right at the edge of the field)
        ref_obj: str
            comma-separated list of filenames (as read from disk, without any
            additional suffixes appended) to be considered object/on-target
            exposures, as overriding guidance for any automatic classification.
        ref_sky: str
            comma-separated list of filenames to be considered as sky exposures

        Any existing OBJFRAME or SKYFRAME flags in the input meta-data will
        also be respected as input (unless overridden by ref_obj/ref_sky) and
        these same keywords are set in the output, along with a group number
        with which each exposure is associated (EXPGROUP).
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]

        # Allow tweaking what size of offset, as a fraction of the field
        # dimensions, is considered to move a target out of the field in
        # gt.group_exposures(). If we want to check this parameter value up
        # front I'm assuming the infrastructure will do that at some point.
        frac_FOV = params["frac_FOV"]

        # Get optional user-specified lists of object or sky filenames, to
        # assist with classifying groups as one or the other. Currently
        # primitive parameters have no concept of lists so we parse a string
        # argument here. As of March 2014 this only works with "reduce2":
        ref_obj = params["ref_obj"].split(',')
        ref_sky = params["ref_sky"].split(',')
        if ref_obj == ['']: ref_obj = []
        if ref_sky == ['']: ref_sky = []

        # Loop over input AstroData objects and extract their filenames, for
        # 2-way comparison with any ref_obj/ref_sky list(s) supplied by the
        # user. This could go into a support function but should become fairly
        # trivial once we address the issues noted below.
        def strip_fits(s):
            return s[:-5] if s.endswith('.fits') else s
        filenames = [strip_fits(ad.phu['ORIGNAME']) for ad in adinputs]

        # Warn the user if they referred to non-existent input file(s):
        missing = [name for name in ref_obj if name not in filenames]
        missing.extend([name for name in ref_sky if name not in filenames])
        if missing:
            log.warning("Failed to find the following file(s), specified "
                "via ref_obj/ref_sky parameters, in the input:")
            for name in missing:
                log.warning("  {}".format(name))

        # Loop over input AstroData objects and apply any overriding sky/object
        # classifications based on user-supplied or guiding information:
        for ad, base_name in zip(adinputs, filenames):
            # Remember any pre-existing classifications so we can point them
            # out if the user requests something different this time:
            obj = ad.phu.get('OBJFRAME') is not None
            sky = ad.phu.get('SKYFRAME') is not None

            # If the user specified manually that this file is object and/or
            # sky, note that in the metadata (alongside any existing flags):
            if base_name in ref_obj and not obj:
                if sky:
                    log.warning("{} previously classified as SKY; added "
                      "OBJECT as requested".format(base_name))
                ad.phu.set('OBJFRAME', 'TRUE')

            if base_name in ref_sky and not sky:
                if obj:
                    log.warning("{} previously classified as OBJECT; added "
                      "SKY as requested".format(base_name))
                ad.phu.set('SKYFRAME', 'TRUE')

            # If the exposure is unguided, classify it as sky unless the
            # user has specified otherwise (in which case we loudly point
            # out the anomaly but do as requested).
            #
            # Although this descriptor naming may not always be ideal for
            # non-Gemini applications, it is our AstroData convention for
            # determining whether an exposure is guided or not.
            if ad.wavefront_sensor() is None:
                # Old Gemini data are missing the guiding keywords but
                # the descriptor then returns None, which is indistinguishable
                # from an unguided exposure (Trac #416). Even phu_get_key_value
                # fails to make this distinction. Whatever we do here (eg.
                # using dates to figure out whether the keywords should be
                # present), it is bound to be Gemini-specific until the
                # behaviour of descriptors is fixed.
                if ('PWFS1_ST' in ad.phu and 'PWFS2_ST' in ad.phu and
                   'OIWFS_ST' in ad.phu):

                    # The exposure was really unguided:
                    if ad.phu.get("OBJFRAME"):
                        log.warning("Exp. {} manually flagged as "
                            "on-target but unguided!".format(base_name))
                    else:
                        log.fullinfo("Treating {} as sky since it's unguided".
                                    format(base_name))
                        ad.phu.set('SKYFRAME', 'TRUE')
                # (else can't determine guiding state reliably so ignore it)

        # Analyze the spatial clustering of exposures and attempt to sort them
        # into dither groups around common nod positions.
        #TODO: The 'Gemini' here is the pkgname, which might not do anything
        groups = gt.group_exposures(adinputs, 'Gemini', frac_FOV=frac_FOV)
        ngroups = len(groups)
        log.fullinfo("Identified {} group(s) of exposures".format(ngroups))

        # Loop over the nod groups identified above, record which group each
        # exposure belongs to, propagate any already-known classification(s)
        # to other members of the same group and determine whether everything
        # is finally on source and/or sky:
        haveobj = False; havesky = False
        allobj = True; allsky = True
        for num, group in enumerate(groups):
            adlist = group.list()
#            obj = False; sky = False
            for ad in adlist:
                ad.phu['EXPGROUP'] = num
#                if ad.phu.get('OBJFRAME'): obj = True
#                if ad.phu.get('SKYFRAME'): sky = True
#                # if obj and sky: break  # no: need to record all group nums
#            if obj:
#                haveobj = True
#                for ad in adlist:
#                    ad.phu['OBJFRAME'] = 'TRUE'
#            else:
#                allobj = False
#            if sky:
#                havesky = True
#                for ad in adlist:
#                    ad.phu['SKYFRAME'] = 'TRUE'
#            else:
#                allsky = False
            # CJS: This reads more cleanly and should do the same thing
            if any(ad.phu.get('OBJFRAME') for ad in adlist):
                haveobj = True
                for ad in adlist:
                    ad.phu.set('OBJFRAME', 'TRUE')
            else:
                allobj = False
            if any(ad.phu.get('SKYFRAME') for ad in adlist):
                havesky = True
                for ad in adlist:
                    ad.phu.set('SKYFRAME', 'TRUE')
            else:
                allsky = False

        # If we now have object classifications but no sky, or vice versa,
        # make whatever reasonable inferences we can about the others:
        if haveobj and not havesky:
            for ad in adinputs:
                if allobj or not ad.phu.get('OBJFRAME'):
                    ad.phu.set('SKYFRAME', 'TRUE')
        elif havesky and not haveobj:
            for ad in adinputs:
                if allsky or not ad.phu.get('SKYFRAME'):
                    ad.phu.set('OBJFRAME', 'TRUE')

        # If all the exposures are still unclassified at this point, we
        # couldn't decide which groups are which based on user input or guiding
        # so use the distance from the target or failing that assume everything
        # is on source but warn the user about it if there's more than 1 group:
        if not haveobj and not havesky:
            # With 2 groups, the one closer to the target position must be
            # on source and the other is presumably sky. For Gemini data, the
            # former, on-source group should be the one with smaller P/Q.
            # TO DO: Once we update ExposureGroup to use RA & Dec descriptors
            # instead of P & Q, this will need changing to subtract the target
            # RA & Dec explicitly. For non-Gemini data where the target RA/Dec
            # are unknown, we'll have to skip this bit and proceed to assuming
            # everything is on source unless given better information.
            if ngroups == 2:
                log.fullinfo("Treating 1 group as object & 1 as sky, based "
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
                        ad.phu.set(key, "TRUE")

            # For more or fewer than 2 groups, we just have to assume that
            # everything is on target, for lack of better information. With
            # only 1 group, this should be a sound assumption, otherwise we
            # warn the user.
            else:
                if ngroups > 1:
                    log.warning("Unable to determine which of {} detected " \
                        "groups are sky/object -- assuming they are all on " \
                        "target AND usable as sky".format(ngroups))
                else:
                    log.fullinfo("Treating a single group as both object & sky")
                for ad in adinputs:
                    ad.phu.set('OBJFRAME', 'TRUE')
                    ad.phu.set('SKYFRAME', 'TRUE')

        # It's still possible for some exposures to be unclassified at this
        # point if the user has identified some but not all of several groups
        # manually (or that's what's in the headers). We can't do anything
        # sensible to rectify that, so just discard the unclassified ones and
        # complain about it.
        missing = [name for ad, name in zip(adinputs, filenames) if
                   not ad.phu.get("OBJFRAME") and not ad.phu.get("SKYFRAME")]
        if missing:
            log.warning("ignoring the following input file(s), which could "
              "not be classified as object or sky after applying incomplete "
              "prior classifications from the input:")
            for name in missing:
                log.warning("  {}".format(name))

        # Construct object & sky lists from the classifications stored above
        # in exposure meta-data, making a complete copy of the input for any
        # duplicate entries (it is hoped this won't require additional memory
        # once memory mapping is used appropriately):
        ad_sci_list = []
        ad_sky_list = []
        for ad in adinputs:
            on_source = ad.phu.get("OBJFRAME")
            if on_source:
                ad_sci_list.append(ad)
            if ad.phu.get("SKYFRAME"):
                if on_source:
                    ad_sky_list.append(deepcopy(ad))
                else:
                    ad_sky_list.append(ad)

        log.stdinfo("Science frames:")
        for ad_sci in ad_sci_list:
            log.stdinfo("  %s" % ad_sci.filename)
        log.stdinfo("Sky frames:")
        for ad_sky in ad_sky_list:
            log.stdinfo("  %s" % ad_sky.filename)

        #TODO: Looks like ad_sci_output_list should become adinputs
        # Add the appropriate time stamp to the PHU and update the filename
        # of the science and sky AstroData objects 
        ad_sci_output_list = gt.finalise_adinput(ad_sci_list,
                    timestamp_key=timestamp_key, suffix=sfx, allow_empty=True)
        ad_sky_output_list = gt.finalise_adinput(ad_sky_list,
                    timestamp_key=timestamp_key, suffix=sfx, allow_empty=True)
               
        # Report the list of output sky AstroData objects to the sky stream in
        # the reduction context
        #rc.report_output(ad_sky_output_list, stream="sky")
        
        # Report the list of output science AstroData objects to the reduction
        # context
        #rc.report_output(ad_sci_output_list)
        adinputs = ad_sci_output_list
        self.streams['sky'] = ad_sky_output_list
        return adinputs

    def skyCorrect(self, adinputs=None, **params):
        #self.scaleSkyToInput()
        adinputs = self.subtractSky(adinputs, **params)
        return adinputs

    def subtractDark(self, adinputs=None, **params):
        """
        This primitive will subtract each SCI extension of the inputs by those
        of the corresponding dark. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the subtraction on the 
        data.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        dark: str/list
            name(s) of the dark file(s) to be subtracted
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        
        dark_list = params["dark"] if params["dark"] else [
            self._get_cal(ad, 'processed_dark') for ad in adinputs]

        # Provide a dark AD object for every science frame
        for ad, dark in zip(*gt.make_lists(adinputs, dark_list,
                                           force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by subtractDark".
                            format(ad.filename))
                continue
            
            if dark is None:
                if 'qa' in self.context:
                    log.warning("No changes will be made to {}, since no "
                                "dark was specified".format(ad.filename))
                    continue
                else:
                    raise IOError("No processed dark listed for {}".
                                   format(ad.filename))

            # Check the inputs have matching filters, binning, and shapes
            try:
                gt.check_inputs_match(ad, dark)
            except ValueError:
                # Else try to extract a matching region from the dark
                dark = gt.clip_auxiliary_data(adinput=ad, aux=dark,
                    aux_type="cal", keyword_comments=self.keyword_comments)

                # Check again, but allow it to fail if they still don't match
                gt.check_inputs_match(ad, dark)

            log.fullinfo("Subtracting the dark ({}) from the input "
                         "AstroData object {}".
                         format(dark.filename, ad.filename))
            ad.subtract(dark)
            
            # Record dark used, timestamp, and update filename
            ad.phu.set('DARKIM', dark.filename, self.keyword_comments["DARKIM"])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
        return adinputs

    def subtractSky(self, adinputs=None, **params):
        """
        This function will subtract the science extension of the input sky
        frames from the science extension of the input science frames. The
        variance and data quality extension will be updated, if they exist.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        sky: list
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        if 'sky' in params and params['sky']:
            # Use the list of sky frames provided by the user. Generate a
            # dictionary associating the input sky AstroData objects to the
            # input science AstroData objects.
            sky_dict = dict(list(zip(**gt.make_lists([ad.phu.get('ORIGNAME')
                                     for ad in adinputs], params['sky']))))

        else:
            # The stackSkyFrames primitive makes the dictionary containing the
            # information associating the stacked sky frames to the science
            # frames an attribute of the PrimitivesClass
            sky_dict = getattr(self, 'stacked_sky_dict')
            
        # Loop over each science AstroData object in the science list
        for ad_sci in adinputs:
            if ad_sci.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by subtractSky".
                            format(ad_sci.filename))
                continue
            
            # Retrieve the sky AstroData object associated with the input
            # science AstroData object
            origname = ad_sci.phu.get("ORIGNAME")
            if origname in sky_dict:
                ad_sky = sky_dict[origname]
                log.stdinfo("Subtracting the sky ({}) from the science "
                            "AstroData object {}".
                            format(ad_sky.filename, ad_sci.filename))
                ad_sci.subtract(ad_sky)

                # Timestamp and update filename
                gt.mark_history(ad_sci, primname=self.myself(),
                                keyword=timestamp_key)
                ad_sci.filename = gt.filename_updater(adinput=ad_sci,
                                        suffix=params["suffix"], strip=True)

            else:
                log.warning("No changes will be made to {}, since no "
                            "appropriate sky could be retrieved".
                            format(ad_sci.filename))

        #TODO: Confirm this is OK. We don't want to keep references to all
        # those sky frames hanging around
        del self.sky_dict

        # Add the appropriate time stamp to the PHU and update the filename
        # of the science and sky AstroData objects 
        #adinputs = gt.finalise_adinput(adinput=adinputs,
        #            timestamp_key=timestamp_key, suffix=params["suffix"])
        return adinputs

    def subtractSkyBackground(self, adinputs=None, **params):
        """
        This primitive is used to subtract the sky background specified by 
        the keyword SKYLEVEL.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by subtractSkyBackground".
                            format(ad.filename))
                continue

            bg_list = ad.hdr.get('SKYLEVEL')
            for ext, bg in zip(ad, bg_list):
                extver = ext.hdr['EXTVER']
                if bg is None:
                    log.warning("No changes will be made to {}:{}, since there "
                                "is no sky background measured".
                                format(ad.filename, extver))
                else:    
                    log.fullinfo("Subtracting {:.0f} to remove sky level from "
                                 "image {}:{}".format(bg, ad.filename, extver))
                    ext.subtract(bg)
                    
            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
        return adinputs

    def thresholdFlatfield(self, adinputs=None, **params):
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
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        lower = params["lower"]
        upper = params["upper"]

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by thresholdFlatfield".
                            format(ad.filename))
                continue

            for ext in ad:
                # Mark the unilumminated pixels with a bit '64' in the DQ plane.
                # make sure the 64 is an int16 64 else it will promote the DQ
                # plane to int64
                unillum = np.where(((ext.data>upper) | (ext.data<lower)) &
                                   (ext.mask & DQ.bad_pixel==0),
                                  np.int16(DQ.unilluminated), np.int16(0))
                ext.mask = unillum if ext.mask is None else ext.mask | unillum
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
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx,
                                              strip=True)
        return adinputs