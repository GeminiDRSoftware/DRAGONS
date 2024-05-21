#
#                                                                  gemini_python
#
#                                                            primitives_image.py
# ------------------------------------------------------------------------------
import numpy as np
from copy import copy, deepcopy
from itertools import product as cart_product

from scipy.ndimage import affine_transform, binary_dilation
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.modeling import models, Model
from gwcs.wcs import WCS as gWCS

import astrodata, gemini_instruments
from astrodata.provenance import add_provenance
from astrodata import Section
from astrodata import wcs as adwcs
from gempy.gemini import gemini_tools as gt
from gempy.library import transform
from geminidr.gemini.lookups import DQ_definitions as DQ
from recipe_system.utils.md5 import md5sum

from .primitives_preprocess import Preprocess
from .primitives_register import Register
from .primitives_resample import Resample
from . import parameters_image

from recipe_system.utils.decorators import parameter_override, capture_provenance


# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class Image(Preprocess, Register, Resample):
    """
    This is the class containing the generic imaging primitives.
    """
    tagset = {"IMAGE"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_image)

    def fringeCorrect(self, adinputs=None, **params):
        """
        Correct science frames for the effects of fringing, using a fringe
        frame. The fringe frame is obtained either from a specified parameter,
        or the "fringe" stream, or the calibration database. This is basically
        a bookkeeping wrapper for subtractFringe(), which does all the work.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        fringe: list/str/AstroData/None
            fringe frame(s) to subtract
        do_fringe: bool/None
            apply fringe correction? (None => use pipeline default for data)
        scale: bool/None
            scale fringe frame? (None => False if fringe frame has same
            group_id() as data
        scale_factor: float/sequence/None
            factor(s) to scale fringe
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        fringe = params["fringe"]
        scale = params["scale"]
        do_cal = params["do_cal"]

        # Exit now if nothing needs a correction, to avoid an error when the
        # calibration search fails. If images with different exposure times
        # are used, some frames may not require a correction (but the calibration
        # search will succeed), so still need to check individual inputs later.
        needs_correction = [self._needs_fringe_correction(ad) for ad in adinputs]
        if any(needs_correction):
            if do_cal == 'skip':
                log.warning("Fringe correction has been turned off but is "
                            "recommended.")
                return adinputs
        else:
            if do_cal == 'procmode' or do_cal == 'skip':
                log.stdinfo("No input images require a fringe correction.")
                return adinputs
            else:  # do_cal == 'force':
                log.warning("Fringe correction has been forced on but may not "
                            "be required.")


        if fringe is None:
            # This logic is for QAP
            try:
                fringe_list = self.streams['fringe']
                assert len(fringe_list) == 1
                scale = False
                log.stdinfo("Using fringe frame in 'fringe' stream. "
                            "Setting scale=False")
                fringe_list = (fringe_list[0], "stream")
            except (KeyError, AssertionError):
                fringe_list = self.caldb.get_processed_fringe(adinputs)
        else:
            fringe_list = (fringe, None)

        # Usual stuff to ensure that we have an iterable of the correct length
        # for the scale factors regardless of what the input is
        scale_factor = params["scale_factor"]
        try:
            factors = iter(scale_factor)
        except TypeError:
            factors = iter([scale_factor] * len(adinputs))
        else:
            # In case a single-element list was passed
            if len(scale_factor) == 1:
                factors = iter(scale_factor * len(adinputs))

        # Get a fringe AD object for every science frame
        for ad, fringe, origin, correct in zip(*gt.make_lists(
                adinputs, *fringe_list, needs_correction, force_ad=(1,))):
            if ad.phu.get(timestamp_key):
                log.warning(f"{ad.filename}: already processed by "
                            "fringeCorrect. Continuing.")
                continue

            # Logic to deal with different exposure times where only
            # some inputs might require fringe correction
            if do_cal == 'procmode' and not correct:
                log.stdinfo("{} does not require a fringe correction".
                            format(ad.filename))
                ad.update_filename(suffix=params["suffix"], strip=True)
                continue

            # At this point, we definitely want to do a fringe correction
            # so we'd better have a fringe frame!
            if fringe is None:
                if 'sq' in self.mode or do_cal == 'force':
                    raise OSError("No processed fringe listed for "
                                  f"{ad.filename}")
                else:
                    log.warning(f"No changes will be made to {ad.filename}, "
                                "since no fringe was specified")
                    continue

            # Check the inputs have matching filters, binning, and shapes
            try:
                gt.check_inputs_match(ad, fringe)
            except ValueError:
                fringe = gt.clip_auxiliary_data(adinput=ad, aux=fringe,
                                                aux_type="cal")
                gt.check_inputs_match(ad, fringe)

            #
            origin_str = f" (obtained from {origin})" if origin else ""
            log.stdinfo(f"{ad.filename}: using the fringe frame "
                         f"{fringe.filename}{origin_str}")
            matched_groups = (ad.group_id() == fringe.group_id())
            if scale or (scale is None and not matched_groups):
                factor = next(factors)
                if factor is None:
                    factor = self._calculate_fringe_scaling(ad, fringe)
                log.stdinfo("Scaling fringe frame by factor {:.3f} before "
                            "subtracting from {}".format(factor, ad.filename))
                # Since all elements of fringe_list might be references to the
                # same AD, need to make a copy before multiplying
                fringe_copy = deepcopy(fringe)
                fringe_copy.multiply(factor)
                ad.subtract(fringe_copy)
            else:
                if scale is None:
                    log.stdinfo("Not scaling fringe frame with same group ID "
                                "as {}".format(ad.filename))
                ad.subtract(fringe)

            # Timestamp and update header and filename
            ad.phu.set("FRINGEIM", fringe.filename, self.keyword_comments["FRINGEIM"])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
            if fringe.path:
                add_provenance(ad, fringe.filename, md5sum(fringe.path) or "", self.myself())
        return adinputs

    def makeFringeForQA(self, adinputs=None, **params):
        """
        Performs the bookkeeping related to the construction of a GMOS fringe
        frame in QA mode.

        The pixel manipulation is left to `makeFringeFrame()`.

        The resulting frame is placed in the "fringe" stream, ready to be
        retrieved by subsequent primitives.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            List of images that must contains at least three elements.

        subtract_median_image : bool
            subtract a median image before finding fringes?

        Other Parameters
        ----------------
        Inherits parameters for `detectSources()` and `stackSkyFrames()`

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Fringe frame. This list contains only one element. The list
            format is maintained so this primitive is consistent with all the
            others.

        See also
        --------
        :meth:`~geminidr.core.primitives_stack.Stack.stackSkyFrames`,
        :meth:`~geminidr.core.primitives_photometry.Photometry.detectSources`
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        # Exit without doing anything if any of the inputs are inappropriate
        if not all(self._needs_fringe_correction(ad) for ad in adinputs):
            return adinputs
        if len({ad.filter_name(pretty=True) for ad in adinputs}) > 1:
            log.warning("Mismatched filters in input; not making fringe frame")
            return adinputs

        # Fringing on Cerro Pachon is generally stronger than on Maunakea.
        # A SExtractor mask alone is usually sufficient for GN data, but GS
        # data need to be median-subtracted to distinguish fringes from objects
        fringe_params = self._inherit_params(params, "makeFringeFrame", pass_suffix=True)
        fringe_adinputs = adinputs

        # Add this frame to the list and get the full list (QA only)
        if "qa" in self.mode:
            # Detect sources in order to get an OBJMASK. Doing it now will aid
            # efficiency by putting the OBJMASK-added images in the list.
            # NB. If we're subtracting the median image, detectSources() has to
            # be run again anyway, so don't do it here.
            # NB2. We don't want to edit adinputs at this stage
            if not fringe_params["subtract_median_image"]:
                fringe_adinputs = [ad if any(hasattr(ext, 'OBJMASK') for ext in ad)
                                   else self.detectSources([ad])[0] for ad in adinputs]
            self.addToList(fringe_adinputs, purpose='forFringe')
            fringe_adinputs = self.getList(purpose='forFringe')

        if len(fringe_adinputs) < 3:
            log.stdinfo("Fewer than 3 frames provided as input. "
                        "Not making fringe frame.")
            return adinputs

        # We have the required inputs to make a fringe frame
        fringe = self.makeFringeFrame(fringe_adinputs, **fringe_params)
        self.streams.update({'fringe': fringe})

        # We now return *all* the input images that required fringe correction
        # so they can all be fringe corrected
        return fringe_adinputs

    def makeFringeFrame(self, adinputs=None, **params):
        """
        Makes a fringe frame from a list of images. This will construct and
        subtract a median image if the fringes are too strong for `detectSources()`
        to work on the inputs as passed. Since a generic recipe cannot know
        whether this parameter is set, including `detectSources()` in the recipe
        prior to making the fringe frame may be a waste of time. Therefore this
        primitive will call `detectSources()` if no `OBJCAT`s are found on the
        inputs.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            List of images that must contains at least three elements.

        suffix : str
            Suffix to be added to output files.

        subtract_median_image : bool
            If True, create and subtract a median image before object detection
            as a first-pass fringe removal.

        Returns
        -------
        adinputs : list of :class:`~astrodata.AstroData`
            Fringe frame. This list contains only one element. The list
            format is maintained so this primitive is consistent with all the
            others.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        min_dist = params.get("debug_distance", 0)

        if len(adinputs) < 3:
            log.stdinfo('Fewer than 3 frames provided as input. '
                        'Not making fringe frame.')
            return []
        coords = [SkyCoord(*ad[0].wcs(0, 0), unit='deg') for ad in adinputs]
        if all([c1.separation(c2).arcsec < min_dist for c1, c2 in
                cart_product(coords, coords)]):
            log.warning("Insufficient dithering, cannot build fringe frame.")
            return []

        adinputs = self.correctBackgroundToReference([deepcopy(ad) for ad in adinputs],
                                            suffix='_bksub', remove_background=True,
                                                     separate_ext=False)

        # If needed, construct a median image and subtract from all frames to
        # do a first-order fringe removal and hence better detect real objects
        if params["subtract_median_image"]:
            median_image = self.stackFrames(adinputs, scale=False,
                            zero=False, operation="median",
                            reject_method="minmax", nlow=0, nhigh=1)
            if len(median_image) > 1:
                raise ValueError("Problem with creating median image")
            median_image = median_image[0]

            # Set the median_image VAR planes to None, or else the adinputs'
            # VAR will increase when subtracting, then adding, the median
            for ext in median_image:
                ext.variance = None

            for ad in adinputs:
                ad.subtract(median_image)
            adinputs = self.detectSources(adinputs,
                        **self._inherit_params(params, "detectSources"))
            for ad in adinputs:
                ad.add(median_image)
        elif not any(hasattr(ext, 'OBJCAT') for ad in adinputs for ext in ad):
            adinputs = self.detectSources(adinputs,
                                              **self._inherit_params(params, "detectSources"))
        else:
            log.stdinfo("OBJCAT found on at least one input extension. "
                        "Not running detectSources.")

        group_ids = {ad.group_id() for ad in adinputs}

        # Add object mask to DQ plane and stack with masking
        # separate_ext is irrelevant unless (scale or zero) but let's be explicit
        adinputs = self.stackSkyFrames(adinputs, mask_objects=True, separate_ext=False,
                                       scale=False, zero=False,
                    **self._inherit_params(params, "stackSkyFrames", pass_suffix=True))
        if len(adinputs) > 1:
            raise ValueError("Problem with stacking fringe frames")

        if len(group_ids) > 1:
            log.stdinfo("Input frames come from more than one group_id. "
                        "Editing fringe frame to avoid matching to science frames.")
            adinputs[0].phu['OBSID'] = '+'.join(group_ids)

        return adinputs

    def resampleToCommonFrame(self, adinputs=None, **params):
        """
        This primitive applies the transformation encoded in the input images
        WCSs to align them with a reference image, in reference image pixel
        coordinates. The reference image is taken to be the first image in
        the input list if not explicitly provided as a parameter.

        By default, the transformation into the reference frame is done via
        interpolation. The variance plane, if present, is transformed in
        the same way as the science data.

        The data quality plane, if present, is handled in a bitwise manner
        with each bit of each pixel in the output image being set it it has
        >1% influence from that bit of a bad pixel. The transformed masks are
        then added back together to generate the transformed DQ plane.

        The WCS objects of the output images are updated to reflect the
        transformation.

        Parameters
        ----------
        suffix : str
            suffix to be added to output files
        interpolant : str
            type of interpolant
        trim_data : bool
            trim image to size of reference image?
        clean_data : bool
            replace bad pixels with a ring median of their values to avoid
            ringing if using a high-order interpolation?
        conserve : bool
            conserve flux when resampling to a different pixel scale?
        force_affine : bool
            convert the true resampling transformation to an affine
            approximation? This speeds up the calculation and has a negligible
            effect for instruments lacking significant distortion
        reference : str/AstroData/None
            reference image for resampling (if not provided, the first image
            in the list will be used)
        dq_threshold : float
            The fraction of a pixel's contribution from a DQ-flagged pixel to
            be considered 'bad' and also flagged.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params.pop("suffix")
        reference = params.pop("reference")
        trim_data = params.pop("trim_data")
        force_affine = params.pop("force_affine")
        # These two parameters are only for GSAOI and will help to define
        # the output WCS if there's no reference image
        pixel_scale = params.pop("pixel_scale", None)
        position_angle = params.pop("pa", None)

        # TODO: Can we make it so that we don't need to mosaic detectors
        # before doing this? That would mean we only do one interpolation,
        # not two, and that's definitely better!
        if not all(len(ad) == 1 or ad.instrument() == "GSAOI" for ad in adinputs):
            raise OSError("All input images must have only one extension.")

        if isinstance(reference, str):
            reference = astrodata.open(reference)
        elif reference is None and pixel_scale is None:
            # Reference image will be the first AD, so we need 2+
            if len(adinputs) < 2:
                log.warning("No alignment will be performed, since at least "
                            "two input AstroData objects are required for "
                            "resampleToCommonFrame")
                return adinputs

        if reference is None and pixel_scale:
            # This must be GSAOI projecting to the requested geometry
            ad0 = adinputs[0]
            ra, dec = ad0.target_ra(), ad0.target_dec()
            # using SkyCoord facilitates formatting the log
            center = SkyCoord(ra * u.deg, dec * u.deg)
            ra_str = center.ra.to_string(u.hour, precision=3)
            dec_str = center.dec.to_string(u.deg, precision=2, alwayssign=True)
            log.stdinfo(f"Projecting with center {ra_str} {dec_str}\n"
                        f"at PA={position_angle} with pixel scale={pixel_scale}")
            pixel_scale /= 3600
            new_wcs = (models.Scale(-pixel_scale) & models.Scale(pixel_scale) |
                       models.Rotation2D(position_angle) |
                       models.Pix2Sky_TAN() |
                       models.RotateNative2Celestial(ra, dec, 180))
            ref_wcs = gWCS([(ad0[0].wcs.input_frame, new_wcs),
                            (ad0[0].wcs.output_frame, None)])
            if trim_data:
                log.warning("Setting trim_data=False as required when no "
                            "reference image is provided.")
                trim_data = False
        else:
            if reference is None:
                reference = adinputs[0]
            else:
                log.stdinfo(f"Using {reference.filename} as reference image")
                if not trim_data:
                    log.warning("Setting trim_data=True to trim to size of the "
                                "reference image.")
                    trim_data = True
            if len(reference) != 1:
                raise OSError("Reference image must have only one extension.")
            ref_wcs = reference[0].wcs

        if trim_data:
            params.update({'origin': (0,) * len(reference[0].shape),
                           'output_shape': reference[0].shape})

        # No transform for the reference AD
        for ad in adinputs:
            transforms = []
            if reference is ad:
                transforms.append(models.Identity(len(ad[0].shape)))
            else:
                for ext in ad:
                    t_align = ext.wcs.forward_transform | ref_wcs.backward_transform
                    if force_affine:
                        affine = adwcs.calculate_affine_matrices(t_align, ext.shape)
                        t_align = models.AffineTransformation2D(matrix=affine.matrix[::-1, ::-1],
                                                                translation=affine.offset[::-1])
                    transforms.append(t_align)

            for ext, t_align in zip(ad, transforms):
                resampled_frame = copy(ext.wcs.input_frame)
                resampled_frame.name = "resampled"
                ext.wcs = gWCS([(ext.wcs.input_frame, t_align),
                                (resampled_frame, ref_wcs.pipeline[0].transform)] +
                                 ref_wcs.pipeline[1:])

        adoutputs = self._resample_to_new_frame(adinputs, frame="resampled",
                                                process_objcat=True, **params)
        for ad in adoutputs:
            try:
                trans_data = ad.nddata[0].meta.pop('transform')
            except KeyError:
                pass
            else:
                corners = np.array(trans_data['corners'][0])
                ncorners = len(corners)
                ad.hdr["AREATYPE"] = (f"P{ncorners}",
                                      f"Region with {ncorners} vertices")
                for i, corner in enumerate(zip(*corners), start=1):
                    for axis, value in enumerate(reversed(corner), start=1):
                        key_name = f"AREA{i}_{axis}"
                        key_comment = f"Vertex {i}, dimension {axis}"
                        ad.hdr[key_name] = (value + 1, key_comment)
                jfactor = trans_data['jfactors'][0]
                ad.hdr["JFACTOR"] = (jfactor, "J-factor in resampling")

            ad.update_filename(suffix=sfx, strip=True)
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)

        return adoutputs

    def scaleByIntensity(self, adinputs=None, **params):
        """
        This primitive scales the inputs so they have the same intensity as
        the reference input (first in the list), which is untouched. Scaling
        can be done by mean or median and a statistics section can be used.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        scaling: str ["mean"/"median"]
            type of scaling to use
        section: str/None
            section of image to use for statistics "x1:x2,y1:y2"
        separate_ext: bool
            if True, scale extensions independently?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        scaling = params["scaling"]
        section = params["section"]
        separate_ext = params["separate_ext"]

        if len(adinputs) < 2:
            log.stdinfo("Scaling has no effect when there are fewer than two inputs")
            return adinputs

        # Do some housekeeping to handle mutually exclusive parameter inputs
        if separate_ext and len({len(ad) for ad in adinputs}) > 1:
            log.warning("Scaling by extension requested but inputs have "
                        "different sizes. Turning off.")
            separate_ext = False

        if section is not None:
            section = Section.from_string(section)

        # I'm not making the assumption that all extensions are the same shape
        # This makes things more complicated, but more general
        targets = [np.nan] * len(adinputs[0])
        for ad in adinputs:
            all_data = []
            for index, ext in enumerate(ad):
                if section is None:
                    _slice = None
                else:
                    _slice = section.asslice()
                data = ext.data[_slice]
                if data.size:
                    mask = None if ext.mask is None else ext.mask[_slice]
                else:
                    log.warning("Section does not intersect with data for "
                                f"{ad.filename} extension {ext.id}."
                                " Using full frame.")
                    data = ext.data
                    mask = ext.mask
                if mask is not None:
                    data = data[mask == 0]

                if not separate_ext:
                    all_data.extend(data.ravel())

                if separate_ext or index == len(ad)-1:
                    if separate_ext:
                        value = getattr(np, scaling)(data)
                        log.fullinfo(f"{ad.filename} extension {ext.id} has "
                                     f"{scaling} value of {value}")
                    else:
                        value = getattr(np, scaling)(all_data)
                        log.fullinfo(f"{ad.filename} has {scaling} value of {value}")

                    if np.isnan(targets[index]):
                        targets[index] = value
                    else:
                        factor = targets[index] / value
                        log.fullinfo("Multiplying by {}".format(factor))
                        if separate_ext:
                            ext *= factor
                        else:
                            ad *= factor

            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs

    def transferObjectMask(self, adinputs=None, **params):
        """
        This primitive takes an image with an OBJMASK and transforms that
        OBJMASK onto the pixel planes of the input images, using their WCS
        information. If the first image is a stack, this allows us to mask
        fainter objects than can be detected in the individual input images.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        source: str
            name of stream containing single stacked image
        interpolant : str
            type of interpolation
        threshold: float
            threshold above which an interpolated pixel should be flagged
        dilation: float
            amount by which to dilate the OBJMASK after transference
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        source = params["source"]
        interpolant = params["interpolant"]
        threshold = params["threshold"]
        dilation = params["dilation"]
        sfx = params["suffix"]

        try:
            source_stream = self.streams[source]
        except KeyError:
            try:
                ad_source = astrodata.open(source)
            except:
                log.warning(f"Cannot find stream or file named {source}. Continuing.")
                return adinputs
        else:
            if len(source_stream) != 1:
                log.warning(f"Stream {source} does not contain single "
                            "AstroData object. Continuing.")
                return adinputs
            ad_source = source_stream[0]

        xgrid, ygrid = np.mgrid[-int(dilation):int(dilation+1),
                                -int(dilation):int(dilation+1)]
        structure = np.where(xgrid*xgrid+ygrid*ygrid <= dilation*dilation,
                             True, False)

        for ad in adinputs:
            force_affine = ad.instrument() != "GSAOI"
            log.stdinfo(f"Transferring object mask to {ad.filename}")
            for ext in ad:
                if hasattr(ext, 'OBJMASK'):
                    log.warning(f"{ad.filename}:{ext.id} already has an "
                                "OBJMASK that will be overwritten")
                ext.OBJMASK = None
                for source_ext in ad_source:
                    t_align = source_ext.wcs.forward_transform | ext.wcs.backward_transform
                    # This line is needed until gWCS PR#405 is merged
                    t_align.inverse = ext.wcs.forward_transform | source_ext.wcs.backward_transform
                    if force_affine:
                        affine = adwcs.calculate_affine_matrices(t_align.inverse, ad[0].shape)
                        objmask = affine_transform(source_ext.OBJMASK.astype(np.float32),
                                                   affine.matrix, affine.offset,
                                                   output_shape=ext.shape, interpolant=interpolant,
                                                   cval=0)
                    else:
                        objmask = transform.Transform(t_align).apply(
                            source_ext.OBJMASK.astype(np.float32),
                            output_shape=ext.shape, interpolant=interpolant, cval=0)
                    objmask = binary_dilation(np.where(abs(objmask) > threshold, 1, 0).
                                              astype(np.uint8), structure).astype(np.uint8)
                    if ext.OBJMASK is None:
                        ext.OBJMASK = objmask
                    else:
                        ext.OBJMASK |= objmask
                # We will deliberately keep the input image's OBJCAT (if it
                # exists) since this will be required for aligning the inputs.
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def _needs_fringe_correction(self, ad):
        """
        Helper method used by fringeCorrect() to determine whether the passed
        AD requires a fringe correction. By default, it is assumed that any
        frame sent to fringeCorrect() does require such a correction, so the
        top-level method simply returns True.

        Parameters
        ----------
        ad: AstroData
            input AD object

        Returns
        -------
        <bool>: does this image need a correction?
        """
        return True

    def _calculate_fringe_scaling(self, ad, fringe):
        """
        Helper method to determine the amount by which to scale a fringe frame
        before subtracting from a science frame. Returns that factor.

        The scaling is determined by minimizing the pixel-to-pixel variance of
        the output frame. If the science frame is S and the fringe frame F,
        then the scaling factor is x where Var(S-xF) is minimized. Doing some
        maths, it can be shown that

            n*Sum(SF) - Sum(S)*Sum(F)  where n is the number of pixels used
        x = -------------------------  and the sums are performed over the
             n*Sum(FF) - (Sum(F))**2   pixels (only good, non-object pixels)

        This works reasonably well for images that are flat, where the pixel-
        to-pixel variation is driven by the fringes and not large-scale
        variations. GMOSImage has a more specific algorithm that uses control
        pairs marking specific parts of the fringe pattern, and this is
        works better, but requires preparatory work setting up those pairs.

        Parameters
        ----------
        ad: AstroData
            input AD object
        fringe: AstroData
            fringe frame

        Returns
        -------
        <float>: scale factor to match fringe to ad
        """
        sum_sci = 0.
        sum_fringe = 0.
        sum_cross = 0.
        sum_fringesq = 0.
        npix = 0

        for ext, fr_ext in zip(ad, fringe):
            # Mask bad pixels in either DQ, and OBJMASK
            mask = (getattr(ext, 'OBJMASK', None) if ext.mask is None else
                    ext.mask | getattr(ext, 'OBJMASK', 0))
            if mask is None:
                mask = fr_ext.mask
            elif fr_ext.mask is not None:
                mask |= fr_ext.mask
            if mask is None:
                good = np.ones_like(ext, dtype=bool)
            else:
                good = (mask == 0)

            sum_sci += np.sum(ext.data[good])
            sum_fringe += np.sum(fr_ext.data[good])
            sum_cross += np.sum(ext.data[good] * fr_ext.data[good])
            sum_fringesq += np.sum(fr_ext.data[good]**2)
            npix += np.sum(good)

        scaling = (npix * sum_cross - sum_sci * sum_fringe) / (npix * sum_fringesq - sum_fringe**2)
        return scaling

    def flagCosmicRaysByStacking(self, adinputs=None, **params):
        """
        This primitive flags sky pixels that deviate from the median image of
        the input AD frames by some positive multiple of a random background
        noise estimate. Since a random noise model is liable to underestimate
        the variance between images in the presence of seeing variations, any
        pixels containing significant object flux are excluded from this
        masking, by running detectSources on the median image and applying the
        resulting OBJMASK array. Any strongly-outlying values in those pixels
        will have to be dealt with when stacking, where less aggressive
        rejection based on the empirical variance between images can be used.

        This is loosely based on the algorithm used by imcoadd in the Gemini
        IRAF package.

        Parameters
        ----------
        suffix: str
            Suffix to be added to output files.
        hsigma: float
            Difference from the median image, in units of the background noise
            estimate, above which pixels should be flagged as contaminated.
        dilation: int
            Dilation radius for expanding cosmic ray mask, in pixels.

        Returns
        -------
        list of AstroData
            The input AstroData instances with flagged pixels added to the
            `mask` array for each extension using the `DQ.cosmic_ray` bit.

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        # timestamp_key = self.timestamp_keys[self.myself()]

        hsigma = params['hsigma']
        dilation = params['dilation']

        if len(adinputs) < 2:
            log.stdinfo("No cosmic rays will be flagged, since at least "
                        "two images are required for {}".format(self.myself()))
            return adinputs

        # This code is taken from dilateObjectMask; factor it out later.
        xgrid, ygrid = np.mgrid[-int(dilation):int(dilation+1),
                       -int(dilation):int(dilation+1)]
        structure = np.where(xgrid*xgrid+ygrid*ygrid <= dilation*dilation,
                             True, False)

        # All inputs should have an OBJMASK, to avoid flagging pixels within
        # objects. If not, we present a warning but continue anyway.
        if not all(hasattr(ext, 'OBJMASK') for ad in adinputs for ext in ad):
            log.warning("Not all input extensions have an OBJMASK. Results "
                        "may be dubious.")

        median_image = self.stackFrames(adinputs, operation='median',
                                        reject_method='none', zero=True)[0]

        median_image = self.detectSources([median_image],
                    **self._inherit_params(params, "detectSources"))[0]

        for ad in adinputs:

            diff = self.subtractSky([deepcopy(ad)], sky=median_image,
                                    offset_sky=True, scale_sky=False)[0]

            # Background will be close to zero, so we only really need this
            # if there's no VAR; however, the overhead is low and it saves
            # us from repeatedly checking if there is a VAR on each extension
            bg_list = gt.measure_bg_from_image(diff, separate_ext=True)

            # Don't flag pixels that are already bad (and may not be CRs;
            # except those that are just near saturation, unilluminated etc.).
            # Also exclude regions with no data, where the variance is 0. so
            # values are always around the threshold.
            bitmask = DQ.bad_pixel | DQ.no_data

            for ext, diff_ext, (bg, noise, npix) in zip(ad, diff, bg_list):
                # Limiting level for good pixels in the median-subtracted data
                # (bkg should be ~0 after subtracting median image with an offset)
                if ext.variance is not None:
                    noise = np.sqrt(ext.variance)
                threshold = bg + hsigma * noise

                # Accumulate CR detections into the DQ mask(s) of the input/output
                crmask = ((diff_ext.data > threshold) &
                          (diff_ext.mask & bitmask == 0))
                if hasattr(diff_ext, 'OBJMASK'):
                    crmask &= (diff_ext.OBJMASK == 0)
                crmask = binary_dilation(crmask, structure)
                ext.mask |= np.where(crmask, DQ.cosmic_ray, DQ.good)

            ad.update_filename(suffix=params["suffix"], strip=True)

        return adinputs

    def _fields_overlap(self, ad1, ad2, frac_FOV=1.0):
        """
        Checks whether the fields of view of two AD objects overlap
        sufficiently to be considerd part of a single ExposureGroup.
        This method, implemented at the Image level, assumes that both
        AD objects have a single extension and a rectangular FOV.
        Instruments which do not fulfil these criteria should have
        their own methods implemented.

        Parameters
        ----------
        ad1: AstroData
            one of the input AD objects
        ad2: AstroData
            the other input AD object
        frac_FOV: float (0 < frac_FOV <= 1)
            fraction of the field of view for an overlap to be considered. If
            frac_FOV=1, *any* overlap is considered to be OK

        Returns
        -------
        bool: do the fields overlap sufficiently?
        """
        # Note that this is not symmetric in ad1 and ad2 if they have different
        # shapes, but that is not expected to happen.
        pos1 = [0.5 * (length - 1) for length in ad1[0].shape[::-1]]
        pos2 = ad2[0].wcs.invert(*ad1[0].wcs(*pos1))
        return all(abs(p1 - p2) < frac_FOV * length
                   for p1, p2, length in zip(pos1, pos2, ad1[0].shape[::-1]))
