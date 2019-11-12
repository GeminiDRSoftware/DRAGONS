#
#                                                                  gemini_python
#
#                                                       primitives_preprocess.py
# ------------------------------------------------------------------------------
import math
import datetime
import numpy as np
from copy import deepcopy
from scipy.ndimage import binary_dilation, filters
from astropy.table import Table
from astropy.convolution import convolve

import astrodata
import gemini_instruments

from gempy.gemini import gemini_tools as gt
from geminidr.gemini.lookups import DQ_definitions as DQ

from geminidr import PrimitivesBASE
from . import parameters_preprocess

from recipe_system.utils.decorators import parameter_override

#import os, psutil
#def memusage(proc):
#    return '{:9.3f}'.format(float(proc.memory_info().rss) / 1000000)
# ------------------------------------------------------------------------------
@parameter_override
class Preprocess(PrimitivesBASE):
    """
    This is the class containing all of the preprocessing primitives.

    """
    tagset = None

    def __init__(self, adinputs, **kwargs):
        super(Preprocess, self).__init__(adinputs, **kwargs)
        self._param_update(parameters_preprocess)

    def addObjectMaskToDQ(self, adinputs=None, suffix=None):
        """
        Combines the object mask in a `OBJMASK` extension into the `DQ` (Data
        Quality) plane.

        Parameters
        ----------
        adinputs : :class:`~astrodata.AstroData`
            Images that contain `OBJMASK`. If `OBJMASK` does not exist, the
            extension is untouched.

        suffix: str/None
            Suffix to be added to output filenames.

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Images with updated `DQ` plane.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

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
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def ADUToElectrons(self, adinputs=None, suffix=None):
        """
        This primitive will convert the units of the pixel data extensions
        of the input AstroData object from ADU to electrons by multiplying
        by the gain.

        Parameters
        ----------
        suffix: str/None
            suffix to be added to output filenames
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

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
            ad.update_filename(suffix=suffix,  strip=True)
        return adinputs
    
    def applyDQPlane(self, adinputs=None, **params):
        """
        This primitive sets the value of pixels in the science plane according
        to flags from the DQ plane. A uniform mean/median or specific value can
        be given, or a ring filter can be used (if inner_radius and outer_radius
        are both defined, and replace_value is *not* a number).

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        replace_flags: int
            The DQ bits, of which one needs to be set for a pixel to be replaced
        replace_value: str/float
            "median" or "mean" to replace with that value of the good pixels,
            or a value
        inner_radius: float/None
            inner radius of the mean/median cleaning filter
        outer_radius: float/None
            outer radius of the cleaning filter
        max_iters: int
            maximum number of cleaning iterations to perform
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        replace_flags = params["replace_flags"]
        replace_value = params["replace_value"]
        inner_radius = params["inner"]
        outer_radius = params["outer"]
        max_iters = params["max_iters"]
        footprint = None

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

                # We need to know the dimensionality of the data to create the
                # footprint but, if we've done it once we can avoid creating
                # it again if the dimensionality of this extension is the same
                if inner_radius is not None and outer_radius is not None:
                    ndim = len(ext.shape)
                    if footprint is None or footprint.ndim != ndim:
                        size = int(outer_radius)
                        mgrid = np.array(np.meshgrid(*([np.arange(-size, size+1)] * ndim)))
                        mgrid *= mgrid
                        footprint = np.sqrt(np.sum(mgrid, axis=0))
                        footprint = np.where(np.logical_and(footprint>=inner_radius,
                                                            footprint<=outer_radius), 1, 0)

                try:
                    rep_value = float(replace_value)
                    log.fullinfo("Replacing bad pixels in {}:{} with the "
                                 "user value {}".format(ad.filename,
                                                        ext.hdr['EXTVER'], rep_value))
                except ValueError:  # already validated so must be "mean" or "median"
                    if footprint is not None:
                        mask = (ext.mask & replace_flags) > 0
                        filtered_data = ext.data
                        iter = 0
                        while (iter < max_iters and np.any(mask)):
                            iter += 1
                            if replace_value == "median":
                                median_data = filters.median_filter(filtered_data, footprint=footprint)
                                filtered_data = np.where(mask, median_data, filtered_data)
                                # If we're median filtering, we can update the mask...
                                # if more than half the input pixels were bad, the
                                # output is still bad.
                                if iter < max_iters:
                                    mask = filters.median_filter(mask, footprint=footprint)
                            else:
                                # "Mean" filtering is just convolution. The astropy
                                # version handles the mask.
                                median_data = convolve(filtered_data, footprint,
                                                       mask=mask, boundary="extend")
                                filtered_data = np.where(mask, median_data, filtered_data)
                                # Output pixels are only bad if *all* the pixels in
                                # the kernel were bad.
                                if iter < max_iters:
                                    mask = np.where(convolve(mask, footprint,
                                                    boundary="extend")>0.9999, True, False)
                        ext.data = filtered_data
                        continue
                    else:
                        oper = getattr(np, replace_value)
                        rep_value = oper(ext.data[ext.mask & replace_flags == 0])
                        log.fullinfo("Replacing bad pixels in {}:{} with the {} "
                                     "of the good data".format(ad.filename,
                                                ext.hdr['EXTVER'], replace_value))

                # kernel-based replacement avoids this line
                ext.data[(ext.mask & replace_flags) != 0] = rep_value

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs

    def associateSky(self, adinputs=None, **params):
        """
        This primitive determines which sky AstroData objects are associated
        with each science AstroData object and puts this information in a
        Table attached to each science frame.

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
            use everything in the "sky" stream?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        min_skies = params["min_skies"]
        max_skies = params["max_skies"]
        min_distsq = params.get("distance", 0) ** 2

        # Create a timedelta object using the value of the "time" parameter
        seconds = datetime.timedelta(seconds=params["time"])

        if params.get('sky'):
            sky = params['sky']
            # Produce a list of AD objects from the sky frame/list
            ad_skies = sky if isinstance(sky, list) else [sky]
            ad_skies = [ad if isinstance(ad, astrodata.AstroData) else
                           astrodata.open(ad) for ad in ad_skies]
        else:  # get from sky stream (put there by separateSky)
            ad_skies = self.streams.get('sky', [])

        # Timestamp and update filenames. Do now so filenames agree at end
        for ad in set(adinputs + ad_skies):
            ad.update_filename(suffix=sfx, strip=True)
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)

        if not adinputs or not ad_skies:
            log.warning("Cannot associate sky frames, since at least one "
                        "science AstroData object and one sky AstroData "
                        "object are required for associateSky")
        else:
            # Create a dict with the observation times to aid in association
            # Allows us to select suitable skies and propagate their datetimes
            sky_times = dict(zip(ad_skies,
                                 [ad.ut_datetime() for ad in ad_skies]))

            for ad in adinputs:
                # If use_all is True, use all of the sky AstroData objects for
                # each science AstroData object
                if params["use_all"]:
                    log.stdinfo("Associating all available sky AstroData "
                                 "objects with {}" .format(ad.filename))
                    sky_list = ad_skies
                else:
                    sci_time = ad.ut_datetime()
                    xoffset = ad.telescope_x_offset()
                    yoffset = ad.telescope_y_offset()

                    # First, select only skies with matching configurations
                    # and within the specified time and with sufficiently
                    # large separation. Keep dict format
                    sky_dict = {k: v for k, v in sky_times.items() if
                                gt.matching_inst_config(ad1=ad, ad2=k,
                                                        check_exposure=True)
                                and ((k.telescope_x_offset() - xoffset)**2 +
                                     (k.telescope_y_offset() - yoffset)**2
                                     > min_distsq)}

                    # Sort sky list by time difference and determine how many
                    # skies will be matched by the default conditions
                    sky_list = sorted(sky_dict, key=lambda x:
                                      abs(sky_dict[x]-sci_time))[:max_skies]
                    num_matching_skies = len([k for k in sky_dict
                                              if abs(sky_dict[k]-sci_time)
                                                 <= seconds])

                    # Now create a sky list of the appropriate length
                    num_skies = min(max_skies or len(sky_list),
                                    max(min_skies or 0, num_matching_skies))
                    sky_list = sky_list[:num_skies]

                    # Sort sky list chronologically for presentation purposes
                    sky_list = sorted(sky_list,
                                      key=lambda sky: sky.ut_datetime())

                if sky_list:
                    sky_table = Table(names=('SKYNAME',),
                                    data=[[sky.filename for sky in sky_list]])
                    log.stdinfo("The sky frames associated with {} are:".
                                 format(ad.filename))
                    for sky in sky_list:
                        log.stdinfo("  {}".format(sky.filename))
                    ad.SKYTABLE = sky_table
                else:
                    log.warning("No sky frames available for {}".format(ad.filename))

        # Need to update sky stream in case it came from the "sky" parameter
        self.streams['sky'] = ad_skies
        return adinputs

    def correctBackgroundToReference(self, adinputs=None, suffix=None,
                                     separate_ext=True, remove_background=False):
        """
        This primitive does an additive correction to a set
        of images to put their sky background at the same level
        as the reference image before stacking.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        remove_background: bool
            if True, set the new background level to zero in all images
            if False, set it to the level of the first image
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        if len(adinputs) <= 1:
            log.warning("No correction will be performed, since at least "
                        "two input AstroData objects are required for "
                        "correctBackgroundToReference")
        # Check that all images have the same number of extensions
        elif not all(len(ad)==len(adinputs[0]) for ad in adinputs):
            raise IOError("Number of science extensions in input "
                                    "images do not match")
        else:
            # Loop over input files
            ref_bg_list = None
            for ad in adinputs:
                bg_list = gt.measure_bg_from_image(ad, value_only=True,
                                                   separate_ext=separate_ext)
                # If this is the first (reference) image, set the reference bg levels
                if ref_bg_list is None:
                    if remove_background:
                        ref_bg_list = ([0] * len(ad)) if separate_ext else 0.
                    else:
                        ref_bg_list = bg_list

                if separate_ext:
                    for ext, bg, ref in zip(ad, bg_list, ref_bg_list):
                        if bg is None:
                            log.warning("Could not get background level from "
                                        "{}:{}".format(ad.filename, ext.hdr['EXTVER']))
                            continue

                        # Add the appropriate value to this extension
                        log.fullinfo("Background level is {:.0f} for {}:{}".
                                     format(bg, ad.filename, ext.hdr['EXTVER']))
                        difference = np.float32(ref - bg)
                        log.fullinfo("Adding {:.0f} to match reference background "
                                     "level {:.0f}".format(difference, ref))
                        ext.add(difference)
                        ext.hdr.set('SKYLEVEL', ref,
                                    self.keyword_comments["SKYLEVEL"])
                else:
                    if bg_list is None:
                        log.warning("Could not get background level from "
                                    "{}".format(ad.filename))
                        continue

                    # Add the appropriate value to the entire AD object
                    log.fullinfo("Background level is {:.0f} for {}".
                                 format(bg_list, ad.filename))
                    difference = np.float32(ref_bg_list - bg_list)
                    log.fullinfo("Adding {:.0f} to match reference background "
                                 "level {:.0f}".format(difference, ref_bg_list))
                    ad.add(difference)
                    ad.hdr.set('SKYLEVEL', ref_bg_list,
                                self.keyword_comments["SKYLEVEL"])

                # Timestamp the header and update the filename
                gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
                ad.update_filename(suffix=suffix, strip=True)

        return adinputs

    def darkCorrect(self, adinputs=None, suffix=None, dark=None, do_dark=True):
        """
        This primitive will subtract each SCI extension of the inputs by those
        of the corresponding dark. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the subtraction on the 
        data. If no dark is provided, getProcessedDark will be called to
        ensure a dark exists for every adinput.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        dark: str/list
            name(s) of the dark file(s) to be subtracted
        do_dark: bool
            perform dark correction?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        if not do_dark:
            log.warning("Dark correction has been turned off.")
            return adinputs

        if dark is None:
            self.getProcessedDark(adinputs, refresh=False)
            dark_list = self._get_cal(adinputs, 'processed_dark')
        else:
            dark_list = dark

        # Provide a dark AD object for every science frame
        for ad, dark in zip(*gt.make_lists(adinputs, dark_list,
                                           force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by darkCorrect".
                            format(ad.filename))
                continue

            if dark is None:
                if 'qa' in self.mode:
                    log.warning("No changes will be made to {}, since no "
                                "dark was specified".format(ad.filename))
                    continue
                else:
                    raise IOError("No processed dark listed for {}".
                                   format(ad.filename))

            # Check the inputs have matching binning, shapes & units
            # TODO: Check exposure time?
            try:
                gt.check_inputs_match(ad, dark, check_filter=False,
                                      check_units=True)
            except ValueError:
                # Else try to extract a matching region from the dark
                dark = gt.clip_auxiliary_data(ad, aux=dark, aux_type="cal")

                # Check again, but allow it to fail if they still don't match
                gt.check_inputs_match(ad, dark, check_filter=False,
                                      check_units=True)

            log.fullinfo("Subtracting the dark ({}) from the input "
                         "AstroData object {}".
                         format(dark.filename, ad.filename))
            ad.subtract(dark)

            # Record dark used, timestamp, and update filename
            ad.phu.set('DARKIM', dark.filename, self.keyword_comments["DARKIM"])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def dilateObjectMask(self, adinputs=None, suffix=None, dilation=1, repeat=False):
        """
        Grows the influence of objects detected by dilating the OBJMASK using
        the binary_dilation routine
        
        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        dilation: float
            radius of dilation circle
        repeat: bool
            allow a repeated dilation? Unless set, the primitive will no-op
            if the appropriate header keyword timestamp is found
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        # Nothing is going to happen so leave now!
        if dilation < 1:
            return adinputs

        xgrid, ygrid = np.mgrid[-int(dilation):int(dilation+1),
                       -int(dilation):int(dilation+1)]
        structure = np.where(xgrid*xgrid+ygrid*ygrid <= dilation*dilation,
                             True, False)

        for ad in adinputs:
            if timestamp_key in ad.phu and not repeat:
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by dilateObjectMask".
                            format(ad.filename))
                continue
            for ext in ad:
                if hasattr(ext, 'OBJMASK') and ext.OBJMASK is not None:
                    ext.OBJMASK = binary_dilation(ext.OBJMASK,
                                                  structure).astype(np.uint8)

            ad.update_filename(suffix=suffix, strip=True)
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
        return adinputs

    def flatCorrect(self, adinputs=None, suffix=None, flat=None, do_flat=True):
        """
        This primitive will divide each SCI extension of the inputs by those
        of the corresponding flat. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the division on the data.
        If no flatfield is provided, getProcessedFlat will be called
        to ensure a flat exists for every adinput.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        flat: str
            name of flatfield to use
        do_flat: bool
            perform flatfield correction?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        if not do_flat:
            log.warning("Flat correction has been turned off.")
            return adinputs

        if flat is None:
            self.getProcessedFlat(adinputs, refresh=False)
            flat_list = self._get_cal(adinputs, 'processed_flat')
        else:
            flat_list = flat

        # Provide a flatfield AD object for every science frame
        for ad, flat in zip(*gt.make_lists(adinputs, flat_list,
                                           force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by flatCorrect".
                            format(ad.filename))
                continue

            if flat is None:
                if 'qa' in self.mode:
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
                flat = gt.clip_auxiliary_data(adinput=ad,
                                    aux=flat, aux_type="cal")
                # Check again, but allow it to fail if they still don't match
                gt.check_inputs_match(ad, flat)

            # Do the division
            log.fullinfo("Dividing the input AstroData object {} by this "
                         "flat:\n{}".format(ad.filename, flat.filename))
            ad.divide(flat)

            # Update the header and filename
            ad.phu.set("FLATIM", flat.filename, self.keyword_comments["FLATIM"])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def makeSky(self, adinputs=None, **params):
        adinputs = self.separateSky(adinputs, **self._inherit_params(params, "separateSky"))
        adinputs = self.associateSky(adinputs, **self._inherit_params(params, "associateSky"))
        #adinputs = self.stackSkyFrames(adinputs, **self._inherit_params(params, "stackSkyFrames"))
        #self.makeMaskedSky()
        return adinputs

    def nonlinearityCorrect(self, adinputs=None, suffix=None):
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

            in_adu = ad.is_in_adu()
            # It's impossible to do this cleverly with a string of ad.mult()s
            # so use regular maths
            log.status("Applying nonlinearity correction to {}".
                       format(ad.filename))
            for ext, coeffs in zip(ad, nonlin_coeffs):
                log.status("   nonlinearity correction for EXTVER {} is {}".
                           format(ext.hdr['EXTVER'], coeffs))
                pixel_data = np.zeros_like(ext.data)

                # Convert back to ADU per exposure if coadds have been summed
                # or if the data have been converted to electrons
                conv_factor = 1 if in_adu else ext.gain()
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
                conv_factor = ext.gain() if in_adu else 1
                if not ext.is_coadds_summed():
                    conv_factor *= ext.coadds()
                if ext.variance is not None and \
                   'poisson' in ext.hdr.get('VARNOISE', '').lower():
                    ext.variance += (pixel_data - ext.data) / conv_factor
                # Now update the SCI extension
                ext.data = pixel_data

            # Timestamp the header and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
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
        operator = getattr(np, params["scale"])

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
            ad.update_filename(suffix=sfx, strip=True)
        return adinputs

    def scaleByExposureTime(self, adinputs=None, **params):
        """
        This primitive scales input images to have the same effective exposure
        time. This can either be provided as a parameter, or the images will be
        scaled to match the exposure time of the first image in the input list.

        Parameters
        ----------
        suffix: str/None
            suffix to be added to output files
        time: float/None
            exposure time to scale to (None => use first image's exposure time)
        """
        log = self.log
        log.debug(gt.log_message("primitive", "scaleByExposureTime", "starting"))
        timestamp_key = self.timestamp_keys["scaleByExposureTime"]
        sfx = params["suffix"]
        time = params["time"]

        # First check if any scaling is actually required
        exptimes = [ad.exposure_time() for ad in adinputs]
        if len(set(exptimes)) == 1 and (time is None or time == exptimes[0]):
            if time is None:
                log.stdinfo("Exposure times are the same therefore no scaling"
                            " is required.")
            else:
                log.stdinfo("Exposure times are all equal to the requested "
                            "time of {}".format(time))
        else:
            for ad, exptime in zip(adinputs, exptimes):
                kw_exptime = ad._keyword_for('exposure_time')
                if time is None:
                    time = exptime
                    log.stdinfo("Scaling to {}'s exposure time of {}".
                                format(ad.filename, time))
                else:
                    scale = time / exptime
                    if abs(scale - 1.0) > 0.001:
                        log.stdinfo("Scaling {} by factor {:.3f}".
                                    format(ad.filename, scale))
                        ad.phu.set(kw_exptime, time,
                                   comment=self.keyword_comments[kw_exptime])
                        # ORIGTEXP should always be the *original* exposure
                        # time, so if it already exists, leave it alone!
                        if "ORIGTEXP" not in ad.phu:
                            ad.phu.set("ORIGTEXP", exptime, "Original exposure time")

                        ad.multiply(scale)
                    else:
                        log.stdinfo("{} does not require scaling".format(ad.filename))

                # Timestamp and update the filename
                gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
                ad.update_filename(suffix=sfx, strip=True)
        return adinputs

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

        # Primitive will construct sets of object and sky frames. First look
        # for pre-assigned header keywords (user can set them as a guide)
        objects = set(filter(lambda ad: 'OBJFRAME' in ad.phu, adinputs))
        skies = set(filter(lambda ad: 'SKYFRAME' in ad.phu, adinputs))

        # Next use optional parameters. These are likely to be passed as
        # comma-separated lists, but should also cope with NoneTypes
        ref_obj = (params["ref_obj"] or '').split(',')
        ref_sky = (params["ref_sky"] or '').split(',')
        if ref_obj == ['']: ref_obj = []
        if ref_sky == ['']: ref_sky = []

        # Add these to the object/sky sets, warning of conflicts
        # use "in" for filename comparison so user can specify rootname only
        def strip_fits(s):
            return s[:-5] if s.endswith('.fits') else s

        missing = []
        for ad in adinputs:
            for obj_filename in ref_obj:
                if strip_fits(obj_filename) in ad.filename:
                    objects.add(ad)
                    if 'SKYFRAME' in ad.phu and 'OBJFRAME' not in ad.phu:
                        log.warning("{} previously classified as SKY; added "
                                "OBJECT as requested".format(ad.filename))
                    break
                missing.append(obj_filename)

            for sky_filename in ref_sky:
                if strip_fits(sky_filename) in ad.filename:
                    objects.add(ad)
                    if 'OBJFRAME' in ad.phu and 'SKYFRAME' not in ad.phu:
                        log.warning("{} previously classified as OBJECT; "
                                "added SKY as requested".format(ad.filename))
                    break
                missing.append(sky_filename)

            # Mark unguided exposures as skies
            if ad.wavefront_sensor() is None:
                # Old Gemini data are missing the guiding keywords and the
                # descriptor returns None. So look to see if the keywords
                # exist; if so, it really is unguided.
                if ('PWFS1_ST' in ad.phu and 'PWFS2_ST' in ad.phu and
                   'OIWFS_ST' in ad.phu):
                    if ad in objects:
                        # Warn user but keep manual assignment
                        log.warning("{} manually flagged as OBJECT but it's "
                                    "unguided!".format(ad.filename))
                    elif ad not in skies:
                        log.fullinfo("Treating {} as SKY since it's unguided".
                                    format(ad.filename))
                        skies.add(ad)
                # (else can't determine guiding state reliably so ignore it)

        # Warn the user if they referred to non-existent input file(s):
        if missing:
            log.warning("Failed to find the following file(s), specified "
                "via ref_obj/ref_sky parameters, in the input:")
            for name in missing:
                log.warning("  {}".format(name))

        # Analyze the spatial clustering of exposures and attempt to sort them
        # into dither groups around common nod positions.
        groups = gt.group_exposures(adinputs, self.inst_lookups, frac_FOV=frac_FOV)
        ngroups = len(groups)
        log.fullinfo("Identified {} group(s) of exposures".format(ngroups))

        # Loop over the nod groups identified above, record which group each
        # exposure belongs to, propagate any already-known classification(s)
        # to other members of the same group and determine whether everything
        # is finally on source and/or sky:
        for num, group in enumerate(groups):
            adlist = group.list()
            for ad in adlist:
                ad.phu['EXPGROUP'] = num

            # If any of these is already an OBJECT, then they all are:
            if objects.intersection(adlist):
                objects.update(adlist)

            # And ditto for SKY:
            if skies.intersection(adlist):
                skies.update(adlist)

        # If one set is empty, try to fill it. Put unassigned inputs in the
        # empty set. If all inputs are assigned, put them all in the empty set.
        if objects and not skies:
            skies = (set(adinputs) - objects) or objects.copy()
        elif skies and not objects:
            objects = (set(adinputs) - skies) or skies.copy()

        # If all the exposures are still unclassified at this point, we
        # couldn't decide which groups are which based on user input or guiding
        # so try to use the distance from the target
        if not objects and not skies:
            if ngroups < 2:  # Includes zero if adinputs=[]
                log.fullinfo("Treating a single group as both object and sky")
                objects = set(adinputs)
                skies = set(adinputs)
            else:
                distsq = [sum([x * x for x in g.group_cen]) for g in groups]
                if ngroups == 2:
                    log.fullinfo("Treating 1 group as object and 1 as sky, "
                                 "based on target proximity")
                    closest = np.argmin(distsq)
                    objects = set(groups[closest].list())
                    skies = set(adinputs) - objects
                else:  # More than 2 groups
                    # Add groups by proximity until at least half the inputs
                    # are classified as objects
                    log.fullinfo("Classifying groups based on target "
                                 "proximity and observation efficiency")
                    for group in [groups[i] for i in np.argsort(distsq)]:
                        objects.update(group.list())
                        if len(objects) >= len(adinputs) // 2:
                            break
                    # We might have everything become an object here, in
                    # which case, make them all skies too (better ideas?)
                    skies = (set(adinputs) - objects) or objects

        # It's still possible for some exposures to be unclassified at this
        # point if the user has identified some but not all of several groups
        # manually (or that's what's in the headers). We can't do anything
        # sensible to rectify that, so just discard the unclassified ones and
        # complain about it.
        missing = [ad for ad in adinputs if ad not in objects | skies]
        if missing:
            log.warning("Ignoring the following input file(s), which could "
              "not be classified as object or sky after applying incomplete "
              "prior classifications from the input:")
            for ad in missing:
                log.warning("  {}".format(ad.filename))

        # Construct object & sky lists (preserving order in adinputs) from
        # the classifications, making a complete copy of the input for any
        # duplicate entries:
        ad_objects = [ad for ad in adinputs if ad in objects]
        ad_skies = [ad for ad in adinputs if ad in skies]
        #ad_skies = [deepcopy(ad) if ad in objects else ad for ad in ad_skies]

        log.stdinfo("Science frames:")
        for ad in ad_objects:
            log.stdinfo("  {}".format(ad.filename))
            ad.phu['OBJFRAME'] = 'TRUE'

        log.stdinfo("Sky frames:")
        for ad in ad_skies:
            log.stdinfo("  {}".format(ad.filename))
            ad.phu['SKYFRAME'] = 'TRUE'

        # Timestamp and update filename for all object/sky frames
        for ad in ad_objects + ad_skies:
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        # Put skies in sky stream and return the objects
        self.streams['sky'] = ad_skies
        return ad_objects

    def skyCorrect(self, adinputs=None, **params):
        """
        This primitive subtracts a sky frame from each of the science inputs.
        Each science input should have a list of skies in a SKYTABLE extension
        and these are stacked and subtracted, using the appropriate primitives.
        
        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        apply_dq: bool
            apply DQ mask to data before combining?
        statsec: str/None
            region of image to use for statistics
        operation: str
            type of combining operation for stacking sky frames
        reject_method: str
            type of rejection method for stacking sky frames
        mask_objects: bool
            mask objects using OBJMASK?
        dilation: float
            dilation radius if objects are being masked
        hsigma: float
            high rejection threshold (standard deviations)
        lsigma: float
            low rejection threshold (standard deviations)
        mclip: bool
            use median (rather than mean) for sigma-clipping?
        nlow: int
            number of low pixels to reject (for "minmax")
        nhigh: int
            number of high pixels to reject (for "minmax")
        memory: float/None
            available memory (in GB) for stacking calculations
        reset_sky: bool
            maintain the sky level by adding a constant to the science
            frame after subtracting the sky?
        scale_sky: bool
            scale each extension of each sky frame to match the science frame?
        offset_sky: bool
            apply offset to each extension of each sky frame to match science?
        sky: str/AD/list
            sky frame(s) to subtract
        """
        #tpid = os.getpid()
        #proc = psutil.Process(tpid)
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        #print "STARTING", memusage(proc)

        save_sky = params["save_sky"]
        reset_sky = params["reset_sky"]
        scale_sky = params["scale_sky"]
        offset_sky = params["offset_sky"]
        if params["scale"] and params["zero"]:
            log.warning("Both the scale and zero parameters are set. "
                        "Setting zero=False.")
            params["zero"] = False

        # Parameters to be passed to stackSkyFrames
        stack_params = self._inherit_params(params, 'stackSkyFrames',
                                            pass_suffix=True)
        #stack_params['mask_objects'] = False  # We're doing this en masse

        # To avoid a crash in certain methods of operation
        if "sky" not in self.streams:
            log.warning("Sky stream is empty. Will search for sky frames in"
                        " main stream.")
            self.streams["sky"] = adinputs

        # We'll need to process the sky frames so collect them all up and do
        # this first, to avoid repeating it every time one is reused
        skies = set()
        skytables = []
        for ad in adinputs:
            try:
                # Sort to ease equality comparisons
                sky_list = sorted(list(ad.SKYTABLE["SKYNAME"]))
                del ad.SKYTABLE  # Not needed any more
            except AttributeError:
                log.warning("{} has no SKYTABLE so cannot subtract a sky "
                            "frame".format(ad.filename))
                sky_list = None
            except KeyError:
                log.warning("Cannot read SKYTABLE associated with {} so "
                            "continuing".format(ad.filename))
                sky_list = None
            skytables.append(sky_list)
            if sky_list:  # Not if None
                skies.update(sky_list)

        # Now make a list of AD instances of the skies, and delete any
        # filenames that could not be converted to ADs
        skies = sorted(list(skies))
        ad_skies = []
        for filename in skies:
            for sky in self.streams["sky"]:
                if sky.filename in [filename,
                        filename.replace(self.params["associateSky"].suffix,
                                         self.params["separateSky"].suffix)]:
                    break
            else:
                try:
                    sky = astrodata.open(filename)
                except astrodata.AstroDataError:
                    log.warning("Cannot find a sky file named {}. "
                            "Ignoring it.".format(filename))
                    skies.remove(filename)
                    continue
                else:
                    log.stdinfo("Found {} on disk".format(filename))
            ad_skies.append(sky)

        # We've got all the sky frames in sky_dict, so delete the sky stream
        # to eliminate references to the original frames before we modify them
        # Note that we can edit the OBJMASK even if the sky is also a science
        # frame because we expect detectSources() to be run again on the
        # sky-subtracted image.
        #del self.streams["sky"]
        if params["mask_objects"]:
            #ad_skies = [ad if any(hasattr(ext, 'OBJMASK') for ext in ad)
            #            else self.detectSources([ad])[0] for ad in ad_skies]
            dilate_params = self._inherit_params(params, "dilateObjectMask")
            ad_skies = self.dilateObjectMask(ad_skies, **dilate_params)
        sky_dict = dict(zip(skies, ad_skies))
        stack_params["dilation"] = 0  # We've already dilated

        # Make a list of stacked sky frames, but use references if the same
        # frames are used for more than one adinput. Use a value "0" to
        # indicate we have not tried to make a sky for this adinput ("None"
        # means we've tried but failed and this can be passed to subtractSky)
        # Fill initial list with None where the SKYTABLE produced None
        stacked_skies = [None if tbl is None else 0 for tbl in skytables]
        for i, (ad, skytable) in enumerate(zip(adinputs, skytables)):
            if skytable is None:
                log.stdinfo("Cannot subtract sky from {}".format(ad.filename))
                continue
            if stacked_skies[i] == 0:
                log.stdinfo("Creating sky frame for {}".format(ad.filename))
                stacked_sky = self.stackSkyFrames([sky_dict[sky] for sky in
                                                  skytable], **stack_params)
                #print ad.filename, memusage(proc)
                if len(stacked_sky) == 1:
                    stacked_sky = stacked_sky[0]
                    # Provide a more intelligent filename
                    stacked_sky.filename = ad.filename
                    stacked_sky.update_filename(suffix="_sky", strip=True)
                else:
                    log.warning("Problem with stacking the following sky "
                                "frames for {}".format(adinputs[i].filename))
                    for filename in skytable:
                        log.warning("  {}".format(filename))
                    stacked_sky = None
                # Assign this stacked sky frame to all adinputs that want it
                for j in range(i, len(skytables)):
                    if skytables[j] == skytable:
                        stacked_skies[j] = stacked_sky
                        if j > i:
                            log.stdinfo("This sky will also be used for {}".format(adinputs[j].filename))
                        skytables[j] = [None]

            # Go through all the science frames and sky-subtract any that
            # aren't needed for future sky-frame creation
            for j, ad2 in enumerate(adinputs):
                # If already been sky-subtracted or not yet processed
                if not skytables[j] or stacked_skies[j] == 0:
                    continue

                # We're iterating over *all* skytables so replace "None"s
                # with iterable empty lists
                if ad2 not in [sky_dict.get(sky) for skytable in skytables
                               for sky in (skytable or [])]:
                    # Sky-subtraction is in place, so we can discard the output
                    self.subtractSky([ad2], sky=stacked_skies[j], scale_sky=scale_sky,
                                     offset_sky=offset_sky, reset_sky=reset_sky,
                                     save_sky=save_sky)
                    skytables[j] = []
                    # This deletes a reference to the AD sky object
                    stacked_skies[j] = None

        # Now we have a list of skies to subtract, one per adinput, so send
        # this to subtractSky as the "sky" parameter
        #print("ABOUT TO SUBTRACT", memusage(proc))
        #adinputs = self.subtractSky(adinputs, sky=stacked_skies, scale_sky=scale_sky,
        #                            offset_sky=offset_sky, reset_sky=reset_sky)
        #print("SUBTRACTED", memusage(proc))
        return adinputs

    def subtractSky(self, adinputs=None, **params):
        """
        This function will subtract the science extension of the input sky
        (or other) frames from the science extension of the input science
        frames. The variance and data quality extension will be updated, if
        they exist.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        reset_sky: bool
            maintain the sky level by adding a constant to the science
            frame after subtracting the sky?
        scale_sky: bool
            scale each extension of each sky frame to match the science frame?
        offset_sky: bool
            apply offset to each extension of each sky frame to match science?
        sky: str/AD/list
            sky frame(s) to subtract
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        save_sky = params["save_sky"]
        reset_sky = params["reset_sky"]
        scale = params["scale_sky"]
        zero = params["offset_sky"]
        if scale and zero:
            log.warning("Both the scale_sky and offset_sky parameters are set. "
                        "Setting offset_sky=False.")
            zero = False

        for ad, ad_sky in zip(*gt.make_lists(adinputs, params["sky"],
                                             force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by subtractSky".
                            format(ad.filename))
                continue

            if ad_sky is not None:
                # Only call measure_bg_from_image if we need it
                if reset_sky or scale or zero:
                    old_bg = gt.measure_bg_from_image(ad, value_only=True)
                log.stdinfo("Subtracting the image ({}) from the science "
                            "AstroData object {}".
                            format(ad_sky.filename, ad.filename))
                if scale or zero:
                    sky_bg = gt.measure_bg_from_image(ad_sky, value_only=True)
                    for ext_sky, final_bg, init_bg in zip(ad_sky, old_bg, sky_bg):
                        if scale:
                            ext_sky *= final_bg / init_bg
                        else:
                            ext_sky += final_bg - init_bg
                        log.fullinfo("Applying {} to EXTVER {} from {} to {}".
                                format(("scaling" if scale else "zeropoint"),
                                       ext_sky.hdr['EXTVER'], init_bg, final_bg))
                if save_sky:
                    #ad_sky.update_filename(suffix='_skyimage', strip=True)
                    self.writeOutputs([ad_sky])
                ad.subtract(ad_sky)
                if reset_sky:
                    new_bg = gt.measure_bg_from_image(ad, value_only=True)
                    for ext, new_level, old_level in zip(ad, new_bg, old_bg):
                        sky_offset = old_level - new_level
                        log.stdinfo("  Adding {} to {}:{}".format(sky_offset,
                                            ad.filename, ext.hdr['EXTVER']))
                        ext.add(sky_offset)
            else:
                log.warning("No changes will be made to {}, since no "
                            "sky was specified".format(ad.filename))

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs

    def subtractSkyBackground(self, adinputs=None, suffix=None):
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
            ad.update_filename(suffix=suffix, strip=True)
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
                if ext.mask is None:
                    ext.mask = np.zeros_like(ext.data, dtype=DQ.datatype)
                # Mark the unilumminated pixels with a bit '64' in the DQ plane.
                # make sure the 64 is an int16(64) else it will promote the DQ
                # plane to int64
                unillum = np.where(((ext.data>upper) | (ext.data<lower)) &
                                   ((ext.mask & DQ.bad_pixel)==0),
                                   np.int16(DQ.unilluminated), np.int16(0))
                ext.mask = unillum if ext.mask is None else ext.mask | unillum
                log.fullinfo("ThresholdFlatfield set bit '64' for values "
                             "outside the range [{:.2f},{:.2f}]".
                             format(lower, upper))

                # Bad pixels might have low values and don't get flagged as
                # unilluminated, so we need to flag them to avoid infinite
                # values in the flatfielded image
                ext.data[ext.data < lower] = 1.0
                ext.data[(ext.mask & DQ.unilluminated)>0] = 1.0
                log.fullinfo("ThresholdFlatfield set flatfield pixels to 1.0 "
                             "for non-illuminated pixels.")

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)
        return adinputs
