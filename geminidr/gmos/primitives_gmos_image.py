from __future__ import division
from __future__ import print_function
#
#                                                                  gemini_python
#
#                                                        primtives_gmos_image.py
#  ------------------------------------------------------------------------------
from builtins import zip
import numpy as np
from copy import deepcopy
import scipy.ndimage as ndimage
from astropy.wcs import WCS

from gempy.gemini import gemini_tools as gt
from gemini_instruments.gmu import detsec_to_pixels

from geminidr.core import Image, Photometry
from .primitives_gmos import GMOS
from . import parameters_gmos_image
from geminidr.gemini.lookups import DQ_definitions as DQ
from geminidr.gmos.lookups.fringe_control_pairs import control_pairs

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class GMOSImage(GMOS, Image, Photometry):
    """
    This is the class containing all of the preprocessing primitives
    for the GMOSImage level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set(["GEMINI", "GMOS", "IMAGE"])

    def __init__(self, adinputs, **kwargs):
        super(GMOSImage, self).__init__(adinputs, **kwargs)
        self._param_update(parameters_gmos_image)

    def addOIWFSToDQ(self, adinputs=None, **params):
        """
        Flags pixels affected by the OIWFS on a GMOS image. It uses the
        header information to determine the location of the guide star, and
        basically "flood-fills" low-value pixels around it to give a first
        estimate. This map is then grown pixel-by-pixel until the values of
        the new pixels it covers stop increasing (indicating it's got to the
        sky level). Extensions to the right of the one with the guide star
        are handled by taking a starting point near the left-hand edge of the
        extension, level with the location at which the probe met the right-
        hand edge of the previous extension.
        
        This code assumes that data_section extends over all rows.
        
        Parameters
        ----------
        border: int
            distance from edge to start flood fill
        convergence: float
            amount within which successive sky level measurements have to
            agree during dilation phase for this phase to finish
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        border = 5  # Pixels in from edge where sky level is reliable
        convergence = 2.0

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

            # OIWFS comes in from the right, so we need to have the extensions
            # sorted in order from left to right
            ampsorder = list(np.argsort([detsec.x1
                                         for detsec in ad.detector_section()]))
            datasec_list = ad.data_section()
            gs_index = -1
            for index in ampsorder:
                ext = ad[index]
                wcs = WCS(ext.hdr)
                x, y = wcs.all_world2pix([[oira, oidec]], 0)[0]
                if x < datasec_list[index].x2 + 0.5:
                    gs_index = index
                    log.fullinfo('Guide star location found at ({:.2f},{:.2f})'
                                 ' on EXTVER {}'.format(x, y, ext.hdr['EXTVER']))
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

            dilator = ndimage.morphology.generate_binary_structure(2, 1)
            for index in ampsorder[amp_index:]:
                datasec = datasec_list[index]
                sky, skysig, _ = gt.measure_bg_from_image(ad[index])

                # To avoid hassle with whether the overscan region is present
                # or not and how adjacent extensions relate to each other,
                # just deal with the data sections
                data_region = ad[index].data[:, datasec.x1:datasec.x2]
                mask_region = ad[index].mask[:, datasec.x1:datasec.x2]
                x1 = max(int(x-border), border)
                x2 = max(min(int(x+border), datasec.x2-datasec.x1), x1+border)
                y1 = max(int(y-border), 0)
                y2 = max(min(int(y+border), datasec.y2-datasec.y1), y1+border)
                wfs_sky = np.median(data_region[y1:y2, x1:x2])
                if wfs_sky > sky-convergence:
                    log.warning('Cannot distinguish probe region from sky for '
                                '{}'.format(ad.filename))
                    break

                # Flood-fill region around guide-star with all pixels fainter
                # than this boundary value
                boundary = sky - 0.2 * (sky-wfs_sky)
                regions, nregions = ndimage.measurements.label(
                    np.logical_and(data_region < boundary, mask_region==0))
                wfs_region = regions[int(y+0.5), int(x+0.5)]
                blocked = ndimage.morphology.binary_fill_holes(np.where(regions==wfs_region,
                                                                        True, False))
                this_mean_sky = wfs_sky
                condition_met = False
                while not condition_met:
                    last_mean_sky = this_mean_sky
                    new_blocked = ndimage.morphology.binary_dilation(blocked,
                                                                     structure=dilator)
                    this_mean_sky = np.median(ad[index].data[new_blocked ^ blocked])
                    blocked = new_blocked
                    if index <= gs_index:
                        condition_met = (this_mean_sky - last_mean_sky < convergence)
                    else:
                        # Dilate until WFS width at left of image equals width at
                        # right of previous extension image
                        width = np.sum(blocked[:,0])
                        condition_met = (y_width - width < 2) or index > 9

                # Flag DQ pixels as unilluminated only if not flagged
                # (to avoid problems with the edge extensions and/or saturation)
                datasec_mask = ad[index].mask[:, datasec.x1:datasec.x2]
                datasec_mask |= np.where(blocked, np.where(datasec_mask>0, 0,
                                                        DQ.unilluminated), 0)

                # Set up for next extension. If flood-fill hasn't reached
                # right-hand edge of detector, stop.
                column = blocked[:, -1]
                y_width = np.sum(column)
                if y_width == 0:
                    break
                y = np.mean(np.arange(datasec.y1, datasec.y2)[column])
                x = border

        return adinputs

#    def fringeCorrect(self, adinputs=None, **params):
#        """
#        This uses a fringe frame to correct a GMOS image for fringing.
#        The fringe frame is obtained either from the calibration database
#        or the "fringe" stream in the reduction (if a suitable file has
#        been constructed using the other primitives in this module).
#
#        CJS: During refactoring, I've changed the operation of this primitive.
#        It used to no-op if *any* of the adinputs didn't need a correction but
#        it now makes an image-by-image decision
#        """
#        log = self.log
#        log.debug(gt.log_message("primitive", self.myself(), "starting"))
#
#        adoutputs = []
#        for ad in adinputs:
#            if self._needs_fringe_correction(ad):
#                # Check for a fringe in the "fringe" stream first; the makeFringe
#                # primitive, if it was called, would have added it there;
#                # this avoids the latency involved in storing and retrieving
#                # a calibration in the central system
#                try:
#                    fringes = self.streams['fringe']
#                    assert len(fringes) == 1
#                except (KeyError, AssertionError):
#                    self.getProcessedFringe([ad])
#                    fringe = self._get_cal(ad, "processed_fringe")
#                    if fringe is None:
#                        log.warning("Could not find an appropriate fringe for {}".
#                                     format(ad.filename))
#                        adoutputs.append(ad)
#                        continue
#
#                    # Scale and subtract fringe
#                    fringe = self.scaleFringeToScience([fringe], science=ad,
#                                                       stats_scale=True)[0]
#                    ad = self.subtractFringe([ad], fringe=fringe)[0]
#                else:
#                    #Fringe was made from science frames. Subtract w/o scaling
#                    log.stdinfo("Using fringe {} for {}".format(
#                        fringes[0].filename, ad.filename))
#                    ad = self.subtractFringe([ad], fringe=fringes)[0]
#
#            adoutputs.append(ad)
#        return adoutputs

    def makeFringe(self, adinputs=None, **params):
        """
        This primitive performs the bookkeeping related to the construction of
        a GMOS fringe frame. The pixel manipulation is left to makeFringeFrame.
        The GMOS version simply handles subtract_median_image=None and then
        calls the Image() version.

        Parameters
        ----------
        subtract_median_image: bool/None
            subtract a median image before finding fringes?
            None => yes if any images are from Gemini-South
        """
        if params["subtract_median_image"] is None:
            params["subtract_median_image"] = any(ad.telescope() == "Gemini-South"
                                                         for ad in adinputs)
        adinputs = super(GMOSImage, self).makeFringe(adinputs, **params)
        return adinputs

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
            ad_for_stats = self.tileArrays([deepcopy(ad)], tile_all=False)[0] \
                if len(ad)>3 else ad

            # Use CCD2, or the entire mosaic if we can't find a second extn
            try:
                ext = ad_for_stats[1]
            except IndexError:
                ext = ad_for_stats[0]

            # Take off 5% of the width as a border
            xborder = max(int(0.05 * ext.data.shape[1]), 20)
            yborder = max(int(0.05 * ext.data.shape[0]), 20)
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
    
    def scaleByIntensity(self, adinputs=None, **params):
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

        ref_mean = None
        for ad in adinputs:
            # If this input hasn't been tiled at all, tile it
            ad_for_stats = self.tileArrays([deepcopy(ad)], tile_all=False)[0] \
                if len(ad)>3 else ad

            # Use CCD2, or the entire mosaic if we can't find a second extn
            try:
                data = ad_for_stats[1].data
            except IndexError:
                data = ad_for_stats[0].data

            # Take off 5% of the width as a border
            xborder = max(int(0.05 * data.shape[1]), 20)
            yborder = max(int(0.05 * data.shape[0]), 20)
            log.fullinfo("Using data section [{}:{},{}:{}] from CCD2 for "
                         "statistics".format(xborder, data.shape[1] - xborder,
                                             yborder, data.shape[0] - yborder))

            stat_region = data[yborder:-yborder, xborder:-xborder]
            mean = np.mean(stat_region)

            # Set reference level to the first image's mean
            if ref_mean is None:
                ref_mean = mean
            scale = ref_mean / mean

            # Log and save the scale factor, and multiply by it
            log.fullinfo("Relative intensity for {}: {:.3f}".format(
                ad.filename, scale))
            ad.phu.set("RELINT", scale,
                                 comment=self.keyword_comments["RELINT"])
            ad.multiply(scale)

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs

    def scaleFringeToScience(self, adinputs=None, **params):
        """
        This primitive will scale the fringes to their matching science data
        The fringes should be in the stream this primitive is called on,
        and the reference science frames should be passed as a parameter.
        
        There are two ways to find the value to scale fringes by:
        1. If stats_scale is set to True, the equation:
        (letting science data = b (or B), and fringe = a (or A))
    
        arrayB = where({where[SCIb < (SCIb.median+2.5*SCIb.std)]} 
                          > [SCIb.median-3*SCIb.std])
        scale = arrayB.std / SCIa.std
    
        The section of the SCI arrays to use for calculating these statistics
        is the CCD2 SCI data excluding the outer 5% pixels on all 4 sides.
        Future enhancement: allow user to choose section
    
        2. If stats_scale=False, then scale will be calculated using:
        exposure time of science / exposure time of fringe

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        stats_scale: bool
            use statistics rather than exposure time to calculate scaling?
        science: list
            list of science frames to scale to
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        science = params["science"]

        if science is None:
            log.warning("No science frames specified; no scaling will be done")
            return adinputs

        fringe_outputs = []
        # We can have multiple science frames but only one fringe
        for ad, fringe in zip(*gt.make_lists(science, adinputs, force_ad=True)):
            # Check the inputs have matching filters, binning and SCI shapes.
            try:
                gt.check_inputs_match(ad, fringe)
            except ValueError:
                # If not, try to clip the fringe frame to the size of the
                # science data and try again
                fringe = gt.clip_auxiliary_data(ad, aux=fringe, aux_type="cal")

                # Check again, but allow it to fail if they still don't match
                gt.check_inputs_match(ad, fringe)

            # Check whether statistics should be used
            stats_scale = params["stats_scale"]

            # Calculate the scale value
            scale = 1.0
            if not stats_scale:
                # Use the exposure times to calculate the scale
                log.fullinfo("Using exposure times to calculate the scaling "
                             "factor")
                try:
                    scale = ad.exposure_time() / fringe.exposure_time()
                except:
                    log.warning("Cannot get exposure times for {} and {}. "
                                "Scaling by statistics instead.".format(
                        ad.filename, fringe.filename))
                    stats_scale = True

            if stats_scale:
                # Use statistics to calculate the scaling factor
                log.fullinfo("Using statistics to calculate the scaling "
                             "factor")

                # Deepcopy the input so it can be manipulated without
                # affecting the original
                ad_for_stats, fringe_for_stats = gt.trim_to_data_section(
                    [deepcopy(ad), deepcopy(fringe)],
                    keyword_comments=self.keyword_comments)

                # CJS: The science and fringe frames should be tiled in the
                # same way before we try to do stats. The old system didn't
                # do proper checking, so let's try to do this properly.

                # First, if one or other is fully tiled, fully tile the other
                if len(ad_for_stats)==1 or len(fringe_for_stats)==1:
                    if len(ad_for_stats) > 1:
                        ad_for_stats = self.tileArrays([ad_for_stats],
                                                       tile_all=True)[0]
                    elif len(fringe_for_stats) > 1:
                        fringe_for_stats = self.tileArrays([fringe_for_stats],
                                                           tile_all=True)[0]
                else:
                    # Tile to the CCD level; tileArrays no-ops if already done
                    # so we can send both frames even if only one needs doing
                    if len(ad_for_stats)>3 or len(fringe_for_stats)>3:
                        ad_for_stats, fringe_for_stats = self.tileArrays(
                            [ad_for_stats,fringe_for_stats], tile_all=False)

                # Use CCD2, or the entire mosaic if we can't find a second extn
                try:
                    sciext = ad_for_stats[1]
                    frngext = fringe_for_stats[1]
                except IndexError:
                    sciext = ad_for_stats[0]
                    frngext = fringe_for_stats[0]

                scidata = sciext.data
                objmask = getattr(sciext, 'OBJMASK', None)
                if sciext.mask is None:
                    dqdata = objmask
                else:
                    dqdata = sciext.mask | objmask \
                        if objmask else sciext.mask
                frngdata = frngext.data

                # Replace any DQ-flagged data with the median value
                if dqdata is not None:
                    smed = np.median(scidata[dqdata==0])
                    scidata = np.where(dqdata!=0, smed, scidata)

                # Calculate the maximum and minimum in a box centered on 
                # each data point.  The local depth of the fringe is
                # max - min.  The overall fringe strength is the median
                # of the local fringe depths.

                # Width of the box is filter dependent, determined by
                # experimentation, but results aren't too heavily affected
                size = 20 if ad.filter_name(pretty=True)=="i" else 40
                size /= ad.detector_x_bin()

                # Use ndimage maximum_filter and minimum_filter to
                # get the local maxima and minima
                sci_max = ndimage.filters.maximum_filter(scidata, size)
                sci_min = ndimage.filters.minimum_filter(scidata, size)

                # Take off 5% of the width as a border
                xborder = max(int(0.05 * scidata.shape[1]), 20)
                yborder = max(int(0.05 * scidata.shape[0]), 20)
                sci_max = sci_max[yborder:-yborder,xborder:-xborder]
                sci_min = sci_min[yborder:-yborder,xborder:-xborder]

                # Take the median difference
                sci_df = np.median(sci_max - sci_min)

                # Do the same for the fringe
                frn_max = ndimage.filters.maximum_filter(frngdata, size)
                frn_min = ndimage.filters.minimum_filter(frngdata, size)
                frn_max = frn_max[yborder:-yborder,xborder:-xborder]
                frn_min = frn_min[yborder:-yborder,xborder:-xborder]
                frn_df = np.median(frn_max - frn_min)

                # This tends to overestimate the factor, but it is
                # at least in the right ballpark, unlike the estimation
                # used in girmfringe (masked_sci.std/fringe.std)
                scale = sci_df / frn_df

            log.fullinfo("Scale factor found = {:.3f}".format(scale))
            scaled_fringe = deepcopy(fringe).multiply(scale)
            
            # Timestamp and update filename
            gt.mark_history(scaled_fringe, primname=self.myself(), keyword=timestamp_key)
            scaled_fringe.filename = ad.filename
            scaled_fringe.update_filename(suffix=params["suffix"], strip=True)
            fringe_outputs.append(scaled_fringe)

        # We're returning the list of scaled fringe frames
        return fringe_outputs
    
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
            log.fullinfo("For {} input frames, using reject_method={}, "
                         "nlow={}, nhigh={}".format(nframes,
                         stack_params["reject_method"], nlow, nhigh))

            # Run the scaleByIntensity primitive to scale flats to the
            # same level, and then stack
            adinputs = self.scaleByIntensity(adinputs)
            adinputs = self.stackFrames(adinputs, **stack_params)
        return adinputs

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
        filter = ad.filter_name(pretty=True)
        exposure = ad.exposure_time()

        if filter not in ["z", "Z", "Y"]:
            log.stdinfo("No fringe correction necessary for {} with filter {}".
                        format(ad.filename, filter))
            return False
        if exposure < 60.0:
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
            return super(GMOSImage, self)._calculate_fringe_scaling(ad, fringe)

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
