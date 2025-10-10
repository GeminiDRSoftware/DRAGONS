#
#                                                                  gemini_python
#
#                                                        primtives_gmos_image.py
#  ------------------------------------------------------------------------------
import numpy as np
from copy import deepcopy
from scipy import ndimage, optimize
from scipy.interpolate import UnivariateSpline

from astropy.modeling import models
from gwcs.wcs import WCS as gWCS
from gwcs import coordinate_frames as cf

from gempy.gemini import gemini_tools as gt
from gempy.library import transform
from gempy.library.nddops import NDStacker
from gemini_instruments.gmu import detsec_to_pixels

from geminidr.core import Image, Photometry
from .primitives_gmos import GMOS
from . import parameters_gmos_image
from .lookups import geometry_conf
from geminidr.gemini.lookups import DQ_definitions as DQ
from geminidr.gmos.lookups.fringe_control_pairs import control_pairs

from recipe_system.utils.decorators import parameter_override, capture_provenance


# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class GMOSImage(GMOS, Image, Photometry):
    """
    This is the class containing all of the preprocessing primitives
    for the GMOSImage level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = {"GEMINI", "GMOS", "IMAGE"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_gmos_image)

    def addOIWFSToDQ(self, adinputs=None, **params):
        """
        Flags pixels affected by the On-Instrument Wavefront Sensor (OIWFS) on a
        GMOS image.

        It uses the header information to determine the location of the
        guide star, and basically "flood-fills" low-value pixels around it to
        give a first estimate. This map is then grown pixel-by-pixel until the
        values of the new pixels it covers stop increasing (indicating it's got to the
        sky level).

        Extensions to the right of the one with the guide star are handled by
        taking a starting point near the left-hand edge of the extension, level
        with the location at which the probe met the right-hand edge of the
        previous extension.

        This code assumes that data_section extends over all rows. It is, of
        course, very GMOS-specific.

        Parameters
        ----------
        adinputs : list of :class:`~gemini_instruments.gmos.AstroDataGmos`
            Science data that contains the shadow of the OIWFS.

        contrast : float (range 0-1)
            Initial fractional decrease from sky level to minimum brightness
            where the OIWFS "edge" is defined.

        convergence : float
            Amount within which successive sky level measurements have to
            agree during dilation phase for this phase to finish.

        Returns
        -------
        list of :class:`~gemini_instruments.gmos.AstroDataGmos`
            Data with updated `.DQ` plane considering the shadow of the OIWFS.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        border = 5  # Pixels in from edge where sky level is reliable
        boxsize = 5
        contrast = params["contrast"]
        convergence = params["convergence"]

        for ad in adinputs:
            wfs = ad.wavefront_sensor()
            if wfs is None or 'OIWFS' not in wfs:
                log.fullinfo('OIWFS not used for image {}.'.format(ad.filename))
                continue

            oira = ad.phu.get('OIARA')
            oidec = ad.phu.get('OIADEC')
            if oira is None or oidec is None:
                log.warning('Cannot determine location of OI probe for {}.'
                            'Continuing.'.format(ad.filename))
                continue

            # DQ planes must exist so the unilluminated region is flagged
            if np.any([ext.mask is None for ext in ad]):
                log.warning('No DQ plane for {}. Continuing.'.format(ad.filename))
                continue

            # OIWFS comes in from the right, so we need to have the extensions
            # sorted in order from left to right
            ampsorder = list(np.argsort([detsec.x1
                                         for detsec in ad.detector_section()]))
            datasec_list = ad.data_section()
            gs_index = -1
            for index in ampsorder:
                ext = ad[index]
                x, y = ext.wcs.backward_transform(oira, oidec)
                if x < datasec_list[index].x2 + 0.5:
                    gs_index = index
                    log.fullinfo(f'Guide star location found at '
                                 f'({x:.2f},{y:.2f}) on extension {ext.id}')
                    break
            if gs_index == -1:
                log.warning('Could not find OI probe location on any extensions.')
                continue

            # The OIWFS extends to the left of the actual star location, which
            # might have it vignetting a part of an earlier extension. Also, it
            # may be in a chip gap, which has the same effect
            amp_index = ampsorder.index(gs_index)
            if x < 50:
                amp_index -= 1
                x = (datasec_list[ampsorder[amp_index]].x2 -
                     datasec_list[ampsorder[amp_index]].x1 - border)
            else:
                x -= datasec_list[ampsorder[amp_index]].x1

            dilator = ndimage.generate_binary_structure(2, 1)
            for index in ampsorder[amp_index:]:
                datasec = datasec_list[index]
                sky, skysig, _ = gt.measure_bg_from_image(ad[index])

                # To avoid hassle with whether the overscan region is present
                # or not and how adjacent extensions relate to each other,
                # just deal with the data sections
                data_region = ad[index].data[:, datasec.x1:datasec.x2]
                mask_region = ad[index].mask[:, datasec.x1:datasec.x2]
                x1 = max(int(x-boxsize), border)
                x2 = max(min(int(x+boxsize), datasec.x2-datasec.x1), x1+border)

                # Try to find the minimum closest to our estimate of the
                # probe location, by downhill method on a spline fit (to
                # smooth out the noise)
                data, mask, var = NDStacker.mean(
                    ad[index].data[:, x1:x2].T, mask=ad[index].mask[:, x1:x2].T)

                good_rows = np.logical_and(mask == DQ.good, var > 0)

                if np.sum(good_rows) == 0:
                    log.warning("No good rows in {} extension {}".format(
                        ad.filename, index))
                    continue

                rows = np.arange(datasec.y2 - datasec.y1)
                spline = UnivariateSpline(rows[good_rows], data[good_rows],
                                          w=1./np.sqrt(var[good_rows]))
                newy = int(optimize.minimize(spline, y, method='CG').x[0] + 0.5)
                y1 = max(int(newy-boxsize), 0)
                y2 = max(min(int(newy+boxsize), len(rows)), y1+border)
                wfs_sky = np.median(data_region[y1:y2, x1:x2])
                if wfs_sky > sky-convergence:
                    log.warning('Cannot distinguish probe region from sky for '
                                '{}'.format(ad.filename))
                    break

                # Flood-fill region around guide-star with all pixels fainter
                # than this boundary value
                boundary = sky - contrast * (sky-wfs_sky)
                regions, nregions = ndimage.label(
                    np.logical_and(data_region < boundary, mask_region==0))
                wfs_region = regions[newy, int(x+0.5)]
                blocked = ndimage.binary_fill_holes(np.where(regions==wfs_region,
                                                             True, False))
                this_mean_sky = wfs_sky
                condition_met = False
                while not condition_met:
                    last_mean_sky = this_mean_sky
                    new_blocked = ndimage.binary_dilation(blocked, structure=dilator)
                    this_mean_sky = np.median(data_region[new_blocked ^ blocked])
                    blocked = new_blocked
                    if index <= gs_index or ad[index].array_section().x1 == 0:
                        # Stop when convergence is reached on either the first
                        # extension looked at, or the leftmost CCD3 extension
                        condition_met = (this_mean_sky - last_mean_sky < convergence)
                    else:
                        # Dilate until WFS width at left of image equals width at
                        # right of previous extension image
                        width = np.sum(blocked[:,0])
                        # Note: this will not be called before y_width is defined
                        condition_met = (y_width - width < 2) or index > 9  # noqa

                # Flag DQ pixels as unilluminated only if not flagged
                # (to avoid problems with the edge extensions and/or saturation)
                datasec_mask = ad[index].mask[:, datasec.x1:datasec.x2]
                datasec_mask |= np.where(blocked, np.where(datasec_mask>0, DQ.good,
                                                        DQ.unilluminated), DQ.good)

                # Set up for next extension. If flood-fill hasn't reached
                # right-hand edge of detector, stop.
                column = blocked[:, -1]
                y_width = np.sum(column)
                if y_width == 0:
                    break
                y = np.mean(np.arange(datasec.y1, datasec.y2)[column])
                x = border

            ad.update_filename(suffix=params["suffix"], strip=True)

        return adinputs

    def applyWCSAdjustment(self, adinputs=None, suffix=None, reference_stream=None):
        """
        This primitive is used when performing separate-CCD reduction of GMOS
        images and adjusts the WCS of the outer CCDs based on images of either
        the complete mosaic or CCD2 only that have already been adjusted.

        The name of a reference stream is supplied that contains images (either
        the full mosaic or CCD2 only) whose WCS has already been corrected. For
        each input AD (either CCD1 or CCD3 only) the WCS is calculated from the
        mosaic geometry and the WCS of the corresponding image in the reference
        stream (determined by matching the data_label). If the reference image
        has been mosaicked, the location of CCD2 is determined from the DQ
        plane, so this primitive will not work if the reference is a mosaic
        without a DQ plane.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        reference_stream: str
            stream containing images which have already been processed by
            adjustWCSToReference
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        try:
            refstream = self.streams[reference_stream]
        except KeyError:
            raise ValueError(f"No stream called {reference_stream}")

        # data_label() produces a unique ID for each observation
        refstream_dict = {ad.data_label(): ad for ad in refstream}

        for ad in adinputs:
            datalab = ad.data_label()
            try:
                ref = refstream_dict[datalab]
            except KeyError:
                log.warning(f"{ad.filename} cannot be processed as no "
                            f"image with data label {datalab} is present "
                            "in the reference stream.")
                continue

            if len(ref) > 1:
                log.warning(f"Reference file {ref.filename} has more than "
                            "one extension. Continuing.")
                continue

            geom = geometry_conf.geometry[ad.detector_name()]
            origins = [k for k in geom if isinstance(k, tuple)]
            ccd2_xorigin = sorted(origins)[1][0]

            # New gWCS objects are going to have the mosaicking transform to
            # map to the frame of CCD2, the mapping from CCD2 to the reference
            # image, and then the reference image's WCS. The reference image
            # might just be CCD2 or it might be a mosaic, in which case we
            # need to know the offset back to the CCD2 frame.
            # We want to rename the input frame here to avoid conflicts.
            ref_wcs = deepcopy(ref[0].wcs)
            ref_wcs = gWCS([(cf.Frame2D(name="reference"),
                             ref_wcs.pipeline[0].transform)]
                           + ref_wcs.pipeline[1:])
            if ref.detector_section()[0].x1 != ccd2_xorigin:
                mask = ref[0].mask
                if mask is None:
                    log.warning(f"Reference file {ref.filename} is not CCD2 "
                                "and does not have a mask. Relative WCS of "
                                f"{ad.filename} will be incorrect.")
                else:
                    xc, yc = mask.shape[1] // 2, mask.shape[0] // 2
                    xorig = xc - np.argmax(mask[yc, xc::-1] & DQ.no_data) + 1
                    yorig = yc - np.argmax(mask.T[xc, yc::-1] & DQ.no_data) + 1
                    if xorig > xc:  # no pixels are NO_DATA
                        xorig = 0
                    if yorig > yc:
                        yorig = 0
                    if xorig != 0 or yorig != 0:
                        shift = models.Shift(xorig) & models.Shift(yorig)
                        ref_wcs.insert_transform(ref_wcs.input_frame, shift, after=True)

            array_info = gt.array_information(ad)
            if len(array_info.origins) != len(ad):
                raise ValueError(f"{ad.filename} has not been tiled")

            transform.add_mosaic_wcs(ad, geometry_conf)

            for ext, origin, detsec in zip(ad, array_info.origins,
                                           ad.detector_section()):
                ext.wcs = gWCS([(ext.wcs.input_frame, ext.wcs.get_transform(ext.wcs.input_frame, 'mosaic'))] +
                               ref_wcs.pipeline)

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)

        return adinputs

    def makeFringeForQA(self, adinputs=None, **params):
        """
        Performs the bookkeeping related to the construction of a GMOS fringe
        frame. The pixel manipulation is left to `makeFringeFrame`.

        The GMOS version simply handles `subtract_median_image=None` and then
        calls the `Image()` version.

        Parameters
        ----------
        adinputs : list of :class:`~gemini_instruments.gmos.AstroDataGmos`
            List of images that must contains at least three elements.

        subtract_median_image : bool or None
            Subtract a median image before finding fringes?
            None => yes if any images are from Gemini-South

        Returns
        -------
        list of :class:`~gemini_instruments.gmos.AstroDataGmos`
            Fringe frame. This list contains only one element. The list
            format is maintained so this primitive is consistent with all the
            others.

        See also
        --------
        :meth:`geminidr.gmos.primitives_gmos_image.GMOSImage.makeFringeFrame`,
        :meth:`geminidr.core.primitives_image.Image.makeFringeFrame`

        """
        params = _modify_fringe_params(adinputs, params)
        return super().makeFringeForQA(adinputs, **params)

    def makeFringeFrame(self, adinputs=None, **params):
        """
        Make a fringe frame from a list of images.

        The GMOS version simply handles `subtract_median_image=None` and then
        calls the `Image()` version.

        Parameters
        ----------
        adinputs : list of :class:`~gemini_instruments.gmos.AstroDataGmos`
            List of images that must contains at least three objects.

        suffix : str
            Suffix to be added to output files.

        subtract_median_image : bool
            If True, create and subtract a median image before object detection
            as a first-pass fringe removal.

        Returns
        -------
        adinputs : list of :class:`~gemini_instruments.gmos.AstroDataGmos`
            Fringe frame. This list contains only one element. The list
            format is maintained so this primitive is consistent with all the
            others.

        See Also
        --------
        :meth:`~geminidr.core.primitives_image.Image.makeFringeFrame`
        """
        params = _modify_fringe_params(adinputs, params)
        return super().makeFringeFrame(adinputs, **params)

    def normalizeFlat(self, adinputs=None, **params):
        """
        This primitive will calculate a normalization factor from statistics
        on CCD2, then divide by this factor and propagate variance accordingly.
        CCD2 is used because of the dome-like shape of the GMOS detector
        response: CCDs 1 and 3 have lower average illumination than CCD2,
        and that needs to be corrected for by the flat.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        for ad in adinputs:
            # If this input hasn't been tiled at all, tile it
            if 'Central Stamp' == ad.detector_roi_setting():
                ad_for_stats = self.tileArrays([deepcopy(ad)], tile_all=False)[0]
                mintrim = 2
            else:
                ad_for_stats = self.tileArrays([deepcopy(ad)], tile_all=False)[0] \
                    if len(ad)>3 else ad
                mintrim = 20

            # Use CCD2, or the entire mosaic if we can't find a second extn
            try:
                ext = ad_for_stats[1]
            except IndexError:
                ext = ad_for_stats[0]

            # Take off 5% of the width as a border
            xborder = max(int(0.05 * ext.data.shape[1]), mintrim)
            yborder = max(int(0.05 * ext.data.shape[0]), mintrim)
            log.fullinfo("Using data section [{}:{},{}:{}] from CCD2 for "
                         "statistics".format(xborder,ext.data.shape[1]-xborder,
                          yborder,ext.data.shape[0]-yborder))

            stat_region = ext.data[yborder:-yborder, xborder:-xborder]

            # Remove DQ-flagged values (including saturated values)
            if ext.mask is not None:
                dqdata = ext.mask[yborder:-yborder, xborder:-xborder]
                stat_region = stat_region[dqdata==0]

            # Remove negative values
            stat_region = stat_region[stat_region>0]

            # Find the mode and standard deviation
            hist,edges = np.histogram(stat_region,
                                      bins=int(np.max(ext.data)/ 0.1))
            mode = edges[np.argmax(hist)]
            std = np.std(stat_region)

            # Find the values within 3 sigma of the mode; the normalization
            # factor is the median of these values
            central_values = stat_region[
                np.logical_and(stat_region > mode - 3 * std,
                               stat_region < mode + 3 * std)]
            norm_factor = np.median(central_values)
            log.fullinfo("Normalization factor: {:.2f}".format(norm_factor))
            ad.divide(norm_factor)

            # Set any DQ-flagged pixels to 1 (to avoid dividing by 0)
            for ext in ad:
                ext.data[ext.mask>0] = 1.0

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs

    def QECorrect(self, adinputs=None, **params):
        """
        This primitive adjusts the background levels of CCDs 1 and 3 with
        multiplicative scalings to match them to the background level of CCD2.
        This is required because of the heterogeneous Hamamatsu CCDs with
        their different QE profiles. The user can provide their own scaling
        factors, or else factors will be calculated by measuring the background
        levels of each CCD.

        Note that this corrects for differences in the *shape* of the QE across
        the imaging filter which cause the relative count rates on each of the
        CCDs to depend on the color of the illumination. Since the twilight sky
        used to flatfield has a different color from the dark night sky of the
        science observations, flatfielding may not return the same count rates
        on all CCDs. The effect is strongest in g.

        This step can effectively be turned off in a recipe by setting the
        parameter factors=1,1

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        factors: list of 2-floats/None
            multiplicative factors to apply to CCDs 1 and 3
            None => calculate from background levels
        common: bool
            if factors is None, calculate and apply the same 2 factors for
            all inputs?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        suffix = params["suffix"]
        factors = params["factors"]
        common = params["common"]
        calc_scaling = not isinstance(factors, list)

        scalings, scaling_samples = [], []
        ads_to_correct = []
        if not calc_scaling:
            log.stdinfo("Using user-supplied scaling factors: "
                        "{:.3f} {:.3f}".format(*factors))
        for ad in adinputs:
            if 'Hamamatsu' not in ad.detector_name(pretty=True):
                log.stdinfo(f"{ad.filename} needs no correction as it does "
                            "not have the Hamamatsu CCDs")
                continue

            array_info = gt.array_information(ad)
            if array_info.detector_shape != (1, 3):
                # TODO? We could use the mask to find the locations of the
                # separate CCDs and still be able to perform this step, and
                # then it could be done after detectSources() on the mosaic
                log.warning(f"{ad.filename} it not comprised of separate CCDs "
                            f"so cannot run {self.myself()} - continuing")
                continue

            bg_measurements = [[], [], []]
            for ccd, extensions in enumerate(array_info.extensions):
                for index in extensions:
                    if calc_scaling:
                        bg, bg_std, nbg = gt.measure_bg_from_image(ad[index])
                        if bg is not None:
                            bg_measurements[ccd].append([bg, nbg])
                    elif ccd != 1:  # should already be validated as 2 elements
                        log.debug(f"Multiplying {ad.filename}:{ad[index].id} by "
                                  f"{factors[ccd // 2]}")
                        ad[index].multiply(factors[ccd // 2])

            if calc_scaling:
                # weight by number of samples in each slice
                bg_measurements = [np.asarray(m, dtype=np.float32)
                                   for m in bg_measurements]
                bg_levels = [np.average(m[:, 0], weights=m[:, 1])
                             for m in bg_measurements]
                log.debug("{} background levels: {:.3f}, {:.3f}, {:.3f}".
                          format(ad.filename, *bg_levels))
                scale_factors = [bg_levels[i] / bg_levels[1] for i in (0, 2)]
                if common:
                    # store scale factors and samples on CCDs 1 and 3
                    scalings.append(scale_factors)
                    scaling_samples.append([bg_measurements[i][:, 1].sum()
                                            for i in (0, 2)])
                    ads_to_correct.append(ad)  # reference for later correction
                else:
                    scale_factors = [1 / factor for factor in scale_factors]
                    log.stdinfo("Calculated scale factors of {:.3f}, {:.3f} for "
                                "{}".format(*scale_factors, ad.filename))
                    for factor, extensions in zip(scale_factors, array_info.extensions[::2]):
                        for index in extensions:
                            log.debug(f"Multiplying {ad.filename}:{ad[index].id} "
                                      f"by {factor}")
                            ad[index].multiply(factor)

            # Timestamp and update filename
            if not (calc_scaling and common):
                gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
                ad.update_filename(suffix=suffix, strip=True)

        # If we had to wait in order to calculate a single pair of scaling
        # factors for all inputs, calculate and apply that now
        if calc_scaling and common and ads_to_correct:
            scale_factors = 1 / np.average(
                np.asarray(scalings), weights=np.asarray(scaling_samples),
                axis=0
            ).astype(np.float32)
            log.stdinfo("Calculated scale factors of {:.3f}, {:.3f} for all "
                        "inputs".format(*scale_factors))
            for ad in ads_to_correct:
                for factor, extensions in zip(
                        scale_factors, gt.array_information(ad).extensions[::2]):
                    for index in extensions:
                        log.debug(f"Multiplying {ad.filename}:{ad[index].id} "
                                  f"by {factor:.3f}")
                        ad[index].multiply(factor)

                # Timestamp and update filename
                gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
                ad.update_filename(suffix=suffix, strip=True)

        return adinputs

    def scaleFlats(self, adinputs=None, **params):
        """
        This primitive scales input images to the mean value of the first
        image. It is intended to be used to scale flats to the same
        level before stacking.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        if len(set(len(ad) for ad in adinputs)) > 1:
            raise ValueError("Not all inputs have the same number of "
                             "extensions")
        if not all(len(set(ad[i].shape for ad in adinputs)) == 1
                   for i in range(len(adinputs[0]))):
            raise ValueError("Not all inputs have extensions with the same "
                             "shapes")

        ref_data_region = None
        for ad in adinputs:
            # If this input hasn't been tiled at all, tile it
            if 'Central Stamp' == ad.detector_roi_setting():
                ad_for_stats = self.tileArrays([deepcopy(ad)], tile_all=False)[0]
                mintrim = 2
            else:
                ad_for_stats = self.tileArrays([deepcopy(ad)], tile_all=False)[0] \
                    if len(ad)>3 else ad
                mintrim = 20

            # Use CCD2, or the entire mosaic if we can't find a second extn
            try:
                data = ad_for_stats[1].data
                mask = ad_for_stats[1].mask
                stat_provider = "CCD2"
            except IndexError:
                data = ad_for_stats[0].data
                mask = ad_for_stats[0].mask
                stat_provider = "mosaic"

            # Take off 5% of the width as a border
            if ref_data_region is None:
                xborder = max(int(0.05 * data.shape[1]), mintrim)
                yborder = max(int(0.05 * data.shape[0]), mintrim)
                log.fullinfo(
                    f"Using data section [{xborder+1}:{data.shape[1]-xborder}"
                    f",{yborder+1}:{data.shape[0]-yborder}] from {stat_provider}"
                    " for statistics")
                region = (slice(yborder, -yborder), slice(xborder, -xborder))
                ref_data_region = data[region]
                ref_mask = mask
                scale = 1
            else:
                data_region = data[region]

                combined_mask = None
                if mask is not None and ref_mask is not None:
                    combined_mask = ref_mask | mask
                elif mask is not None:
                    combined_mask = mask
                elif ref_mask is not None:
                    combined_mask = ref_mask

                if combined_mask is not None:
                    mask_region = combined_mask[region]
                    mean = np.mean(data_region[mask_region == 0])
                    ref_mean = np.mean(ref_data_region[mask_region == 0])
                else:
                    mean = np.mean(data_region)
                    ref_mean = np.mean(ref_data_region)

                # Set reference level to the first image's mean
                scale = ref_mean / mean

            # Log and save the scale factor, and multiply by it
            log.fullinfo(f"Relative intensity for {ad.filename}: {scale:.3f}")
            ad.phu.set("RELINT", scale,
                                 comment=self.keyword_comments["RELINT"])
            ad.multiply(scale)

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs

    def stackFlats(self, adinputs=None, **params):
        """
        This primitive will combine the input flats with rejection
        parameters set appropriately for GMOS imaging twilight flats.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files

        apply_dq: bool
            apply DQ mask to data before combining? (passed to stackFrames)

        operation: str
            type of combine operation (passed to stackFrames)

        reject_method: str
            rejection method (passed to stackFrames)

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        nframes = len(adinputs)
        if nframes < 2:
            log.stdinfo("At least two frames must be provided to stackFlats")
        else:
            # Define rejection parameters based on number of input frames,
            # to be used with minmax rejection. Note: if reject_method
            # parameter is overridden, these parameters will just be
            # ignored
            stack_params = self._inherit_params(params, "stackFrames")
            stack_params["reject_method"] = "minmax"
            nlow, nhigh = 0, 0
            if nframes <= 2:
                stack_params["reject_method"] = "none"
            elif nframes <= 5:
                nlow, nhigh = 1, 1
            elif nframes <= 10:
                nlow, nhigh = 2, 2
            else:
                nlow, nhigh = 2, 3
            stack_params.update({'nlow': nlow, 'nhigh': nhigh,
                                 'zero': False, 'scale': False,
                                 'statsec': None, 'separate_ext': False})
            log.fullinfo(f"For {nframes} input frames, using reject_method="
                         f"{stack_params['reject_method']}, "
                         f"nlow={nlow}, nhigh={nhigh}")

            # Run the scaleByIntensity primitive to scale flats to the
            # same level, and then stack
            adinputs = self.scaleFlats(adinputs)
            adinputs = self.stackFrames(adinputs, **stack_params)
        return adinputs

    def _fields_overlap(self, ad1, ad2, frac_FOV=1.0):
        raise NotImplementedError("GMOSImage has no _fields_overlap() method")

    def _needs_fringe_correction(self, ad):
        """
        This function determines whether an AstroData object needs a fringe
        correction. If it says no, it reports its decision to the log.

        Parameters
        ----------
        ad: AstroData
            input AD object

        Returns
        -------
        <bool>: does this image need a correction?
        """
        log = self.log
        inst = ad.instrument()
        det = ad.detector_name(pretty=True)
        filter = ad.filter_name(pretty=True)
        exposure = ad.exposure_time()

        if filter not in ["z", "Y"] and \
           not (filter in ["i", "CaT", "Z"] and det in ["EEV", "e2vDD"]):
            log.stdinfo("No fringe correction necessary for {} with filter {}".
                        format(ad.filename, filter))
            return False
        # Short QA exposures don't get corrected due to time pressure
        if 'qa' in self.mode and exposure < 60.0:
            log.stdinfo("No fringe correction necessary for {} with "
                        "exposure time {:.1f}s".format(ad.filename, exposure))
            return False
        return True

    def _calculate_fringe_scaling(self, ad, fringe):
        """
        Helper method to determine the amount by which to scale a fringe frame
        before subtracting from a science frame. Returns that factor.

        This uses the method of Snodgrass & Carry (2013; ESO Messenger 152, 14)
        with a series of "control pairs" of locations at the peaks and troughs
        of fringes. The differences between the signals at these pairs are
        calculated for both the science and fringe frames, and the average
        ratio between these is used as the scaling.

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
        log = self.log
        halfsize = 10

        # TODO: Do we have CCD2-only images to defringe?
        detname = ad.detector_name()
        try:
            pairs = control_pairs[detname]
        except KeyError:
            log.warning("Cannot find control pairs for detector {} in {}. "
                        "Using defualt scaling algorithm".format(detname, ad.filename))
            return super()._calculate_fringe_scaling(ad, fringe)

        # Different detectors => different fringe patterns
        if detname != fringe.detector_name():
            log.warning("Input {} and fringe {} appear to have different "
                        "detectors".format(ad.filename, fringe.filename))

        scale_factors = []
        for pair in pairs:
            signals = []
            for image in (ad, fringe):
                for (x, y) in pair:
                    i1, x1, y1 = detsec_to_pixels(image, detx=x-halfsize,
                                                  dety=y-halfsize)
                    i2, x2, y2 = detsec_to_pixels(image, detx=x+halfsize+1,
                                                  dety=y+halfsize+1)
                    if i1 == i2:
                        signals.append(np.median(image[i1].data[y1:y2, x1:x2]))
            if len(signals) == 4:
                scaling = (signals[0] - signals[1]) / (signals[2] - signals[3])
                log.debug("{} produces {}".format(signals, scaling))
                scale_factors.append(scaling)

        if scale_factors:
            if len(scale_factors) < 6:
                log.warning("Only {} control pair measurements made: fringe "
                            "scaling is uncertain".format(len(scale_factors)))
            scaling = np.median(scale_factors)
        else:
            log.warning("Failed to estimate fringe scaling for {}".
                             format(ad.filename))
            scaling = 1.
        return scaling

#-----------------------------------------------------------------------------
def _modify_fringe_params(adinputs, params):
    """
    This function modifies the param dictionary for the makeFringeForQA() and
    makeFringeFrame() primitives, to allow subtract_median_image=None to be
    passed.

    Parameters
    ----------
    adinputs: list
        AD instances being processed
    params: dict
        parameters passed to the calling primitive

    Returns
    -------
    dict: a (possibly modified) version of params
    """
    if params["subtract_median_image"] is None:
        params["subtract_median_image"] = any(ad.telescope() == "Gemini-South"
                                              for ad in adinputs)
    return params
