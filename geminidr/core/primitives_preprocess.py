#
#                                                                  gemini_python
#
#                                                       primitives_preprocess.py
# ------------------------------------------------------------------------------
import datetime
import math
from collections import defaultdict
from copy import deepcopy
from contextlib import suppress
from functools import partial

import astrodata
import gemini_instruments  # noqa
import matplotlib.pyplot as plt
import numpy as np
from astrodata.provenance import add_provenance
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.ndimage import binary_dilation

from geminidr import PrimitivesBASE, CalibrationNotFoundError
from geminidr.gemini.lookups import DQ_definitions as DQ
from gempy.gemini import gemini_tools as gt
from gempy.library import astromodels as am, astrotools as at
from gempy.library import tracing
from gempy.library.filtering import ring_median_filter
from recipe_system.utils.decorators import parameter_override, capture_provenance
from recipe_system.utils.md5 import md5sum

from gempy.utils.errors import ConvergenceError

from . import parameters_preprocess


@parameter_override
@capture_provenance
class Preprocess(PrimitivesBASE):
    """
    This is the class containing all of the preprocessing primitives.

    """
    tagset = None

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
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
                    log.warning(f'No object mask present for {ad.filename} '
                                f'extension {ext.id}; cannot apply object mask')
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def ADUToElectrons(self, adinputs=None, suffix=None):
        """
        This primitive will convert the units of the pixel data extensions
        of the input AstroData object from ADU to electrons by multiplying
        by the gain. The gain keyword in each extension is then set to 1.0
        to represent the new conversion factor.

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

            # Now multiply the pixel data in each science extension by the gain
            # and the pixel data in each variance extension by the gain squared
            log.status("Converting {} from ADU to electrons by multiplying by "
                       "the gain".format(ad.filename))
            for ext in ad:
                if not ext.is_in_adu():
                    log.warning(f"  {ext.id} is already in electrons. "
                                "Continuing.")
                    continue
                gain = gt.array_from_descriptor_value(ext, "gain")
                ext.multiply(np.float32(gain))  # avoid casting int to float64

                # Update saturation and nonlinear levels with new value. We
                # allowed these to return lists before this point but now a
                # single (mean) value is going to be used.
                for desc in ('saturation_level', 'non_linear_level'):
                    try:
                        kw = ad._keyword_for(desc)
                    except AttributeError:
                        continue
                    if kw in ext.hdr:
                        new_value = np.mean(
                            gain * gt.array_from_descriptor_value(ext, desc))
                        # Make sure we update the comment too!
                        new_comment = ext.hdr.comments[kw].replace('ADU', 'electron')
                        ext.hdr[kw] = (new_value, new_comment)

            # Update the headers of the AstroData Object. The pixel data now
            # has units of electrons so update the physical units keyword.
            ad.hdr.set('BUNIT', 'electron', self.keyword_comments['BUNIT'])
            try:
                ad.hdr.set(ad._keyword_for("gain"), 1.)
            except AttributeError:  # No keyword for "gain"
                pass
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

        flag_list = [int(math.pow(2, i))
                     for i, digit in enumerate(bin(replace_flags)[2:][::-1])
                     if digit == '1']
        log.stdinfo(f"The flags {flag_list} will be applied")

        for ad in adinputs:
            for ext in ad:
                if ext.mask is None:
                    log.warning(f"No DQ plane exists for {ad.filename} "
                                f"extension {ext.id}, so the correction "
                                "cannot be applied")
                    continue

                try:
                    rep_value = float(replace_value)
                    log.fullinfo(f"Replacing bad pixels in {ad.filename}"
                                 f"extension {ext.id} with the "
                                 f"user value {rep_value}")
                except ValueError:
                    # If replace_value is a string. It was already validated
                    # so must be "mean" or "median"
                    if inner_radius is not None and outer_radius is not None:
                        ring_median_filter(ext, inner_radius, outer_radius,
                                           max_iters=max_iters, inplace=True,
                                           replace_flags=replace_flags,
                                           replace_func=replace_value)
                        continue
                    else:
                        oper = getattr(np, replace_value)
                        rep_value = oper(ext.data[ext.mask & replace_flags == 0])
                        log.fullinfo(f"Replacing bad pixels in {ad.filename} "
                                     f"extension {ext.id} with the "
                                     f"{replace_value} of the good data")

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
        min_skies: int/None
            minimum number of skies to associate to each input frame
        sky: str/list
            name(s) of sky frame(s) to associate to each input
        time: float
            number of seconds
        use_all: bool
            use all input frames as skies (unless they are too close on the sky)?
        """
        def sky_coord(ad):
            """Return (RA, dec) at center of first extension"""
            return SkyCoord(*ad[0].wcs(*[x // 2 - 0.5
                                         for x in ad[0].shape[::-1]])[-2:], unit='deg')

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        min_skies = params["min_skies"]
        max_skies = params["max_skies"]
        min_dist = params.get("distance", 0)

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

        has_skytable = [False] * len(adinputs)
        if not adinputs or not ad_skies:
            log.warning("Cannot associate sky frames, since at least one "
                        "science AstroData object and one sky AstroData "
                        "object are required for associateSky")
        else:
            # Get the observation times and coordinates of all skies to aid
            # with association. We calculate the coordinates at the center of
            # the first extension, which should cope with different ROIs (but
            # there is a matching_inst_config() call later)
            sky_times = [ad.ut_datetime() for ad in ad_skies]
            sky_coords = {ad: sky_coord(ad) for ad in set(adinputs + ad_skies)}

            for i, ad in enumerate(adinputs):
                coord = sky_coord(ad)
                # If use_all is True, use all of the AstroData objects (sky
                # or object) for each science AstroData object (as long as
                # the separation is sufficient)
                if params["use_all"]:
                    log.stdinfo("Associating all displaced sky AstroData "
                                f"objects with {ad.filename}")
                    sky_list = [ad_other for ad_other in set(adinputs + ad_skies)
                                if coord.separation(sky_coords[ad_other]).arcsec > min_dist]
                else:
                    sci_time = ad.ut_datetime()

                    # First, select only skies with matching configurations
                    # and within the specified time and with sufficiently
                    # large separation. Keep dict format
                    sky_dict = {ad_sky: t for (ad_sky, t) in zip(ad_skies, sky_times) if
                                gt.matching_inst_config(ad1=ad, ad2=ad_sky,
                                                        check_exposure=True)
                                and coord.separation(sky_coords[ad_sky]).arcsec > min_dist}

                    # Sort sky list by time difference and determine how many
                    # skies will be matched by the default conditions
                    sky_list = sorted(sky_dict, key=lambda x:
                                      abs(sky_dict[x] - sci_time))[:max_skies]
                    num_matching_skies = len([k for k in sky_dict
                                              if abs(sky_dict[k] - sci_time) <= seconds])

                    # Now create a sky list of the appropriate length
                    if num_matching_skies < min_skies <= len(sky_dict):
                        log.warning(f"Found fewer skies ({num_matching_skies}) "
                                    "matching the time criterion than requested\n"
                                    f"by 'min_skies' ({min_skies}). "
                                    "Ignoring the time parameter.\n"
                                    "Enforcing min_skies.")
                        log.warning("To enforce the time parameter, set min_skies to 0.")
                    elif num_matching_skies < min_skies:
                        log.warning(f"Found fewer skies ({num_matching_skies}) "
                                    f"than requested by 'min_skies' ({min_skies}).\n"
                                    "Using as many skies as possible.")
                    num_skies = min(max_skies or len(sky_list),
                                    max(min_skies or 0, num_matching_skies))
                    sky_list = sky_list[:num_skies]

                    # Sort sky list chronologically for presentation purposes
                    sky_list = sorted(sky_list,
                                      key=lambda sky: sky.ut_datetime())

                if sky_list:
                    sky_table = Table(names=('SKYNAME',),
                                      data=[[sky.filename for sky in sky_list]])
                    log.stdinfo(f"The sky frames associated with {ad.filename} are:")
                    for sky in sky_list:
                        log.stdinfo(f"  {sky.filename}")
                    ad.SKYTABLE = sky_table
                    has_skytable[i] = True
                    log.stdinfo("")
                else:
                    log.warning(f"No sky frames available for {ad.filename}")

        # Need to update sky stream in case it came from the "sky" parameter
        self.streams['sky'] = ad_skies

        # if none of frames have sky tables, just pass them all through
        # if only some frames did not have sky corrected, move them out of main and
        # to the "no_skytable" stream.
        if not any(has_skytable):  # "all false", none have been sky corrected
            log.warning('Sky frames could not be associated to any input frames. '
                        'Sky subtraction will not be possible.')
        elif not all(has_skytable):  # "some false", some frames were NOT sky corrected
            log.stdinfo('')  # for readablity
            false_idx = [idx for idx, trueval in enumerate(has_skytable) if not trueval]
            for idx in reversed(false_idx):
                ad = adinputs[idx]
                log.warning(f'{ad.filename} does not have any associated skies'
                            ' and cannot be sky-subtracted, moving to '
                            '"no_skies" stream')
                if "no_skies" in self.streams:
                    self.streams["no_skies"].append(ad)
                else:
                    self.streams["no_skies"] = [ad]
                del adinputs[idx]

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
            raise ValueError("Number of science extensions in input images do "
                             "not match")
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
                                        f"{ad.filename} extension {ext.id}")
                            continue

                        # Add the appropriate value to this extension
                        log.fullinfo(f"Background level is {bg:.0f} for "
                                     f"{ad.filename} extension {ext.id}")
                        difference = np.float32(ref - bg)
                        log.fullinfo(f"Adding {difference:.0f} to match "
                                     f"reference background level {ref:.0f}")
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

    def darkCorrect(self, adinputs=None, suffix=None, dark=None, do_cal=None):
        """
        This primitive will subtract each SCI extension of the inputs by those
        of the corresponding dark. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the subtraction on the
        data. If no dark is provided, the calibration database(s) will be
        queried.

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

        if do_cal == 'skip':
            log.warning("Dark correction has been turned off.")
            return adinputs

        if dark is None:
            dark_list = self.caldb.get_processed_dark(adinputs)
        else:
            dark_list = (dark, None)

        # Provide a dark AD object for every science frame, and an origin
        for ad, dark, origin in zip(*gt.make_lists(adinputs, *dark_list,
                                    force_ad=(1,))):
            if ad.phu.get(timestamp_key):
                log.warning(f"{ad.filename}: already processed by "
                            "darkCorrect. Continuing.")
                continue

            if dark is None:
                if 'sq' in self.mode or do_cal == 'force':
                    raise CalibrationNotFoundError("No processed dark listed "
                                                   f"for {ad.filename}")
                else:
                    log.warning(f"No changes will be made to {ad.filename}, "
                                "since no dark was specified")
                    continue

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

            origin_str = f" (obtained from {origin})" if origin else ""
            log.stdinfo(f"{ad.filename}: subtracting the dark "
                        f"{dark.filename}{origin_str}")
            ad.subtract(dark)

            # Record dark used, timestamp, and update filename
            ad.phu.set('DARKIM', dark.filename, self.keyword_comments["DARKIM"])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)

            if dark.path:
                add_provenance(ad, dark.filename, md5sum(dark.path) or "", self.myself())
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


    def fixPixels(self, adinputs=None, **params):
        """
        This primitive replaces bad pixels by linear interpolation along
        lines or columns using the nearest good pixels, similar to IRAF's
        fixpix.

        Regions must be specified either as a string, separated by semi-colons,
        with the ``regions`` parameter, or with a file (``regions_file``), one
        region per line.

        Regions strings must be a comma-separated list of colon-separated
        pixel coordinates or ranges, one per axis, in 1-indexed Cartesian
        pixel co-ordinates, inclusive of the upper limit. Axes are specified
        in Fortran order (reverse of the Python order). The extension can
        be specified at the beginning of the string, separated from the
        coordinates by a slash. If extension is not specified, the region
        will be fixed for all extensions.

        Examples::

            450, 521 => single pixel, line 521, column 450
            430:437, 513:533 => lines 513 to 533, columns 430 to 437
            10:, 100 => line 100, columns 10 to the end
            *, 100 => line 100
            2/429:,100 => for extension 2 only

        By default, interpolation is performed across the narrowest dimension
        spanning bad pixels with interpolation along image lines if the two
        dimensions are equal (in the 2D case). 3D is also supported with the
        same behavior. For single pixels it is possible to use a local median
        filter instead.

        Parameters
        ----------
        adinputs : list of `~astrodata.AstroData`
            List of input files.
        suffix : str
            suffix to be added to output files.
        regions : str
            List of pixels or regions to fix (see description above).
        regions_file : str
            Path to a file containing the regions to fix. If both regions_file
            and regions are supplied, both will be used and regions_file will
            be used first.
        axis : int  [None or 1 - 3]
            Axis over which the interpolation is done, 1 is along the x-axis,
            2 is along the y-axis, 3 is the z-axis.  If None (default), the
            axis is determined from the narrowest dimension of each region.
        use_local_median : bool
            Use a local median filter for single pixels?
        debug : bool
            Display regions?

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        axis = params['axis']
        debug = params['debug']
        regions_file = params['regions_file']
        regions = params['regions']
        suffix = params['suffix']
        use_local_median = params['use_local_median']

        if regions is None and regions_file is None:
            raise ValueError('regions must be specified either as a string '
                             '(regions) or with a file (regions_file)')

        all_regions = []

        if regions_file is not None:
            with open(regions_file) as f:
                all_regions += f.read().strip().splitlines()

        if regions is not None:
            all_regions += regions.split(';')

        region_slices = defaultdict(list)
        for region in all_regions:
            if '/' in region:
                ext, reg = region.split('/')
            else:
                ext, reg = 0, region  # applies to all extensions

            try:
                slices = at.cartesian_regions_to_slices(reg.strip())
            except ValueError:
                log.warning(f'Failed to parse region: {reg}')
            else:
                region_slices[int(ext)].append((region, slices))

        for ad in adinputs:
            for iext, ext in enumerate(ad, start=1):
                ndim = ext.data.ndim

                if ext.mask is None:
                    ext.mask = np.zeros(ext.shape, dtype=DQ.datatype)

                for region, slices in region_slices[0] + region_slices[iext]:
                    if len(slices) != ndim:
                        raise ValueError(f'region {region} does not match '
                                         'array dimension')

                    if ext.data[slices].size == 0:
                        raise ValueError(f'region {region} out of bound')

                    region_shape = np.array([
                        (s.stop or ext.shape[i]) - (s.start or 0)
                        for i, s in enumerate(slices)
                    ])

                    # Find the axis that will be used for the interpolation
                    if axis is None:
                        # If we have two axis with the same size, we should
                        # use the deeper one. E.g. for images with a square
                        # region, interpolation is done on lines.
                        use_axis = np.where(
                            region_shape == region_shape.min())[0][-1]
                    else:
                        if axis not in range(1, ndim + 1):
                            raise ValueError('axis should specify a dimension '
                                             f'between 1 and {ndim}')
                        use_axis = ndim - axis

                    if debug:
                        log.debug(f'Replacing pixel {region} with a '
                                  'local median ')
                        plot_slices = [slice(None if sl.start is None else sl.start - 10,
                                             None if sl.stop is None else sl.stop + 10)
                                       for sl in slices]
                        if len(plot_slices) > 2:
                            plot_slices = [Ellipsis] + plot_slices[-2:]
                        origdata = ext.data[plot_slices].copy()

                    if use_local_median and np.prod(region_shape) == 1:
                        ext.mask[slices] |= 32768
                        ring_median_filter(ext.nddata, 3, 5, inplace=True,
                                           replace_flags=32768,
                                           replace_func='median')
                        ext.mask[slices] ^= 32768
                    else:
                        log.debug(f'Interpolating region {region} on '
                                  f'axis {ndim - use_axis}')
                        # Extract the data corresponding to the region
                        slices_extract = list(slices)
                        slices_extract[use_axis] = slice(None)
                        data = ext.data[tuple(slices_extract)]

                        # Reshape to put the interpolation axis first, and
                        # flatten the other axes
                        data = np.rollaxis(data, use_axis)
                        extracted_shape = data.shape
                        data = data.reshape(ext.shape[use_axis], -1)

                        # Prepare the data to interpolate, removing the values
                        # from the region
                        sl = slices[use_axis]
                        ind = np.arange(ext.shape[use_axis])
                        ind = np.delete(ind, sl)
                        data_in = np.delete(data, sl, axis=0)

                        if ind.size == 0:
                            if axis is None:
                                raise ValueError(
                                    'no good data left for the interpolation')
                            else:
                                raise ValueError(
                                    'no good data left for the interpolation '
                                    'along the chosen axis')

                        # Do the interpolation and replace the values
                        f = interp1d(ind, data_in, kind='linear', axis=0,
                                     bounds_error=True)
                        data[sl, :] = f(np.arange(sl.start, sl.stop))

                        # Reshape the other way, and replace the final values
                        data = data.reshape(extracted_shape)
                        data = np.rollaxis(data, 0, use_axis + 1)
                        ext.data[tuple(slices_extract)] = data

                    # Mark the interpolated pixels as no_data
                    ext.mask[slices] |= DQ.no_data

                    if debug:
                        fig, (ax1, ax2) = plt.subplots(1, 2)
                        ax1.imshow(origdata, vmin=0, vmax=100)
                        ax2.imshow(ext.data[plot_slices], vmin=0, vmax=100)
                        plt.show()

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def flatCorrect(self, adinputs=None, suffix=None, flat=None, do_cal=None):
        """
        This primitive will divide each SCI extension of the inputs by those
        of the corresponding flat. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the division on the data.
        If no flatfield is provided, the calibration database(s) will be
        queried.

        If the flatfield has had a QE correction applied, this information is
        copied into the science header to avoid the correction being applied
        twice.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        flat: str
            name of flatfield to use
        do_cal: str [procmode|force|skip]
            perform flatfield correction?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        qecorr_key = self.timestamp_keys['QECorrect']

        if do_cal == 'skip':
            log.warning("Flat correction has been turned off.")
            return adinputs

        if flat is None:
            flat_list = self.caldb.get_processed_flat(adinputs)
        else:
            flat_list = (flat, None)

        # Provide a bflat AD object for every science frame, and an origin
        for ad, flat, origin in zip(*gt.make_lists(adinputs, *flat_list,
                                    force_ad=(1,))):
            if ad.phu.get(timestamp_key):
                log.warning(f"{ad.filename}: already processed by "
                            "flatCorrect. Continuing.")
                continue

            if flat is None:
                if 'sq' in self.mode or do_cal == 'force':
                   raise CalibrationNotFoundError("No processed flat listed "
                                                  f"for {ad.filename}")
                else:
                   log.warning(f"No changes will be made to {ad.filename}, "
                               "since no flatfield has been specified")
                   continue

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
            origin_str = f" (obtained from {origin})" if origin else ""
            log.stdinfo(f"{ad.filename}: dividing by the flat "
                         f"{flat.filename}{origin_str}")
            ad.divide(flat)

            # Try to get a slit rectification model from the flat, and, if one
            # exists, insert it before the pixels-to-world transform.
            ad = gt.attach_rectification_model(ad, flat, log=self.log)

            # Update the header and filename, copying QECORR keyword from flat
            ad.phu.set("FLATIM", flat.filename, self.keyword_comments["FLATIM"])
            try:
                qecorr_value = flat.phu[qecorr_key]
            except KeyError:
                pass
            else:
                log.fullinfo("Copying {} keyword from flatfield".format(qecorr_key))
                ad.phu.set(qecorr_key, qecorr_value, flat.phu.comments[qecorr_key])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
            if flat.path:
                add_provenance(ad, flat.filename, md5sum(flat.path) or "", self.myself())
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

        def linearize(counts, coeffs):
            """Return a linearized version of the counts in electrons per coadd"""
            ret_counts = np.zeros_like(counts)
            for coeff in reversed(coeffs):
                ret_counts[:] += coeff
                ret_counts[:] *= counts
            return ret_counts

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning(f"No changes will be made to {ad.filename}, since "
                            "it has already been processed by nonlinearityCorrect")
                continue

            # Get the correction coefficients
            try:
                nonlin_coeffs = self._nonlinearity_coeffs(ad)
            except:
                log.warning("Unable to obtain nonlinearity coefficients for "
                            f"{ad.filename}")
                continue

            in_adu = ad.is_in_adu()
            # It's impossible to do this cleverly with a string of ad.mult()s
            # so use regular maths
            log.status(f"Applying nonlinearity correction to {ad.filename}")
            for ext, gain, coeffs in zip(ad, ad.gain(), nonlin_coeffs):
                log.status("   nonlinearity correction for extension "
                           f"{ext.id} is {coeffs}")

                # Ensure we linearize the electrons per exposure
                coadds = ext.coadds() if ext.is_coadds_summed() else 1
                conv_factor = gain / coadds
                pixel_data = linearize(ext.data * conv_factor, coeffs) / conv_factor

                # Try to do something useful with the VAR plane, if it exists
                # Since the data are fairly pristine, VAR will simply be the
                # Poisson noise (divided by gain if in ADU, divided by COADDS
                # if the coadds are averaged), possibly plus read-noise**2
                # So making an additive correction will sort this out,
                # irrespective of whether there's read noise
                if ext.variance is not None and \
                   'poisson' in ext.hdr.get('VARNOISE', '').lower():
                    ext.variance += ((pixel_data - ext.data) /
                                     (gain * (1 if ext.is_coadds_summed() else ext.coadds())))
                # Now update the SCI extension
                ext.data = pixel_data

                for desc in ('saturation_level', 'non_linear_level'):
                    with suppress(AttributeError):
                        current_value = getattr(ext, desc)()
                        new_value = linearize(
                            [current_value * conv_factor], coeffs)[0] / conv_factor
                        ext.hdr[ad._keyword_for(desc)] = np.round(new_value, 3)

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
                    log.fullinfo("Normalizing {} extension {} by dividing by {:.2f}"
                                 .format(ad.filename, ext.id, scaling))
                    ext /= scaling
            else:
                # Combine pixels from all extensions, using DQ if present
                scaling = operator(np.concatenate([
                    (ext.data.ravel() if ext.mask is None else ext.data[ext.mask==0].ravel())
                    for ext in ad])).astype(np.float32)
                log.fullinfo(f"Normalizing {ad.filename} by dividing by {scaling:.2f}")
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

    def scaleCountsToReference(self, adinputs=None, **params):
        """
        This primitive scales the input images so that the scaled fluxes of
        the sources in the OBJCAT match those in the reference image (the
        first image in the list). By setting the input parameter tolerance=0,
        it is possible to simply scale the images by the exposure times.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        tolerance: float (0 <= tolerance <= 1)
            tolerance within which scaling must match exposure time to be used
        use_common: bool
            use only sources common to all frames?
        radius: float
            matching radius in arcseconds
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        tolerance = params["tolerance"]
        use_common = params["use_common"]
        radius = params["radius"]

        if len(adinputs) <= 1:
            log.stdinfo("No scaling will be performed, since at least two "
                        f"AstroData objects are required for {self.myself()}")
            return adinputs

        all_image = all('IMAGE' in ad.tags for ad in adinputs)
        all_spect = all('SPECT' in ad.tags for ad in adinputs)
        if not (all_image ^ all_spect):
            raise ValueError("All inputs must be either IMAGE or SPECT")

        if all_image:
            mkcat = mkcat_image
            calc_scaling = calc_scaling_image
        else:
            all_spect1d = all(len(ext.shape) == 1 for ad in adinputs for ext in ad)
            all_spect2d = all(len(ad) == 1 and len(ad[0].shape) == 2
                              for ad in adinputs)
            if not (all_spect1d ^ all_spect2d):
                raise ValueError("All inputs must either be single-extension "
                                "2D spectra or multi-extension 1D spectra")
            # Spectral extraction in this primitive does not subtract the sky
            if (all_spect2d and tolerance > 0 and not
                    all(self.timestamp_keys['skyCorrectFromSlit'] in ad.phu
                        for ad in adinputs)):
                log.warning("Not all inputs have been sky-corrected. "
                            "Scaling may be in error.")
            mkcat = mkcat_spect
            calc_scaling = calc_scaling_spect

        # extract a SkyCoord object from a catalogue. Annoyingly, the list of
        # RA and DEC must be a *list* and NOT a tuple, so abstract this ugliness
        get_coords = lambda x: SkyCoord(*[list(k) for k in zip(*list(x.keys()))],
                                        unit=u.deg)

        ref_ad = adinputs[0]
        if tolerance > 0:
            #if all_image and set(len(ad) for ad in adinputs) != {1}:
            #    raise ValueError(f"{self.myself()} requires all inputs to have "
            #                     "only 1 extension")
            try:
                ref_objcat = mkcat(ref_ad)
            except ValueError as e:
                log.warning(f"Cannot construct catalogue from reference ({e})"
                            " - continuing")
                return adinputs
            ref_coords = get_coords(ref_objcat)
        else:
            log.stdinfo("Scaling all images by exposure time only.")
            use_common = False  # it's irrelevant

        kw_exptime = ref_ad._keyword_for('exposure_time')
        ref_texp = ref_ad.exposure_time()
        scale_factors = [1]  # for first (reference) image
        nmatched = [0]
        exptimes = [ref_texp]

        # If use_common is True, we'll have two passes through this loop:
        # the first to identify the sources in common to all frames, and the
        # second to do the work. If it's False, we only do the second pass.
        # use_common is False when we want to calculate the scalings.

        # Avoid two passes with only two images, since we'll use all matches
        if len(adinputs) == 2:
            use_common=False

        while True:
            for ad in adinputs[1:]:
                texp = ad.exposure_time()
                exptimes.append(texp)
                time_scaling = ref_texp / texp
                if tolerance == 0:
                    scale_factors.append(time_scaling)
                    nmatched.append(0)
                    continue
                try:
                    objcat = mkcat(ad)
                except ValueError as e:
                    if use_common:
                        log.warning(f"{e} - there will be no objects common "
                                    "to all images. Setting use_common=False.")
                        ref_objcat = mkcat(ref_ad)  # reset
                        break
                    log.warning(f"{e} - scaling by exposure times ({time_scaling})")
                    scale_factors.append(time_scaling)
                    nmatched.append(0)
                    continue

                log.debug(f"{ad.filename} catalog has {len(objcat)} sources")
                coords = get_coords(objcat)
                idx, d2d, _ = ref_coords.match_to_catalog_sky(coords)
                matched = d2d < radius * u.arcsec
                if use_common:  # still making the common source catalogue
                    ref_objcat = {k: v for (k, v), m in zip(ref_objcat.items(), matched) if m}
                    ref_coords = ref_coords[matched]
                    log.debug(f"After {ad.filename} there are {len(ref_objcat)} "
                              "sources in common.")
                    if not ref_objcat:
                        log.warning("No objects are common to all images. "
                                    "Setting use_common=False.")
                        ref_objcat = mkcat(ref_ad)  # reset
                        break
                else:  # calculate the scaling
                    if matched.sum():
                        scaling = calc_scaling(ad, ref_ad, objcat, ref_objcat,
                                               idx, matched, self.log)
                        if (scaling > 0 and (tolerance == 1 or
                                            ((1 - tolerance) <= scaling <= 1 / (1 - tolerance)))):
                            scale_factors.append(scaling)
                            nmatched.append(matched.sum())
                        else:
                            log.warning(f"Scaling factor {scaling:.3f} for "
                                        f"{ad.filename} (from {matched.sum()} "
                                        "sources is inconsisent with exposure "
                                        f"time scaling {time_scaling:.3f}")
                            scale_factors.append(time_scaling)
                            nmatched.append(0)
                    else:
                        log.warning(f"No sources matched between {ref_ad.filename}"
                                    f" and {ad.filename}")
                        scale_factors.append(time_scaling)
            if not use_common:
                break
            use_common = False

        for ad, scaling, exptime, num in zip(adinputs, scale_factors,
                                             exptimes, nmatched):
            if num > 0:
                log.stdinfo(f"Scaling {ad.filename} by {scaling:.3f} "
                            f"(from {num} sources)")
            else:
                log.stdinfo(f"Scaling {ad.filename} by {scaling:.3f}")
            if scaling != 1:
                ad.multiply(scaling)
                # ORIGTEXP should always be the *original* exposure
                # time, so if it already exists, leave it alone!
                if "ORIGTEXP" not in ad.phu:
                    ad.phu.set("ORIGTEXP", exptime, "Original exposure time")
                # The new exposure time should probably be the reference's
                # exposure time, so that all the outputs have the same value
                ad.phu.set(kw_exptime, ref_texp,
                           comment=self.keyword_comments[kw_exptime])
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)

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
        max_perp_offset = params.get("debug_allowable_perpendicular_offset")

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
        for obj_filename in ref_obj:
            for ad in adinputs:
                if strip_fits(obj_filename) in ad.filename:
                    objects.add(ad)
                    if 'SKYFRAME' in ad.phu and 'OBJFRAME' not in ad.phu:
                        log.warning(f"{ad.filename} previously classified as "
                                    "SKY; added OBJECT as requested")
                    break
            else:
                missing.append(obj_filename)

        for sky_filename in ref_sky:
            for ad in adinputs:
                if strip_fits(sky_filename) in ad.filename:
                    skies.add(ad)
                    if 'OBJFRAME' in ad.phu and 'SKYFRAME' not in ad.phu:
                        log.warning(f"{ad.filename} previously classified as "
                                    "OBJECT; added SKY as requested")
                    break
            else:
                missing.append(sky_filename)

        # Analyze the spatial clustering of exposures and attempt to sort them
        # into dither groups around common nod positions.
        if 'SPECT' in adinputs[0].tags:
            overlap_func = partial(self._fields_overlap,
                                   max_perpendicular_offset=max_perp_offset)
        else:
            overlap_func = self._fields_overlap
        groups = gt.group_exposures(adinputs, fields_overlap=overlap_func,
                                    frac_FOV=frac_FOV)
        ngroups = len(groups)
        log.stdinfo(f"Identified {ngroups} group(s) of exposures")

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

        for ad in adinputs:
            # Mark unguided exposures as skies
            if ad.wavefront_sensor() is None:
                # Old Gemini data are missing the guiding keywords and the
                # descriptor returns None. So look to see if the keywords
                # exist; if so, it really is unguided.
                if ('PWFS1_ST' in ad.phu and 'PWFS2_ST' in ad.phu and
                        'OIWFS_ST' in ad.phu):
                    if ad in objects:
                        # Warn user but keep manual assignment
                        log.warning(f"{ad.filename} manually flagged as "
                                    f"OBJECT but it's unguided!")
                    elif ad not in skies:
                        log.stdinfo(f"Treating {ad.filename} as SKY since "
                                    "it's unguided")
                        skies.add(ad)
                # (else can't determine guiding state reliably so ignore it)

        # Warn the user if they referred to non-existent input file(s):
        if missing:
            log.warning("Failed to find the following file(s), specified "
                        "via ref_obj/ref_sky parameters, in the input:")
            for name in missing:
                log.warning(f"  {name}")

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
                log.stdinfo("Treating a single group as both object and sky")
                objects = set(adinputs)
                skies = set(adinputs)
            else:
                # Not all ADs necessarily have the same target coords, but
                # we have to pick a single target location, so use the first
                target = SkyCoord(adinputs[0].target_ra(),
                                  adinputs[0].target_dec(), unit=u.deg)
                dist = [target.separation(group.group_center).arcsec
                        for group in groups]
                if ngroups == 2:
                    log.stdinfo("Treating 1 group as object and 1 as sky, "
                                "based on target proximity")
                    closest = np.argmin(dist)
                    objects = set(groups[closest].members)
                    skies = set(adinputs) - objects
                else:  # More than 2 groups
                    # Add groups by proximity until at least half the inputs
                    # are classified as objects
                    log.stdinfo("Classifying groups based on target "
                                "proximity and observation efficiency")
                    for group in [groups[i] for i in np.argsort(dist)]:
                        objects.update(group.members)
                        if len(objects) >= len(adinputs) / 2:
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
                log.warning(f"  {ad.filename}")

        # Construct object & sky lists (preserving order in adinputs) from
        # the classifications, making a complete copy of the input for any
        # duplicate entries:
        ad_objects = [ad for ad in adinputs if ad in objects]
        ad_skies = [ad for ad in adinputs if ad in skies]
        #ad_skies = [deepcopy(ad) if ad in objects else ad for ad in ad_skies]

        log.stdinfo("Science frames:")
        for ad in ad_objects:
            log.stdinfo(f"  {ad.filename}")
            ad.phu['OBJFRAME'] = 'TRUE'

        log.stdinfo("Sky frames:")
        for ad in ad_skies:
            log.stdinfo(f"  {ad.filename}")
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
        scale: bool
            scale the sky frames before stacking them?
        zero: bool
            apply offets to the sky frames before stacking them?
        reset_sky: bool
            maintain the sky level by adding a constant to the science
            frame after subtracting the sky?
        scale_sky: bool
            scale each extension of each stacked sky frame to match the
            science frame?
        offset_sky: bool
            apply offset to each extension of each stacked sky frame to match
            the science frame?
        sky: str/AD/list
            sky frame(s) to subtract
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        save_sky = params["save_sky"]
        reset_sky = params["reset_sky"]
        scale_sky = params.get("scale_sky", False)
        offset_sky = params.get("offset_sky", False)
        suffix = params["suffix"]
        if "zero" not in params.keys():
            params["zero"] = False
        if "scale" not in params.keys():
            params["scale"] = False
        if params["scale"] and params["zero"]:
            log.warning("Both the scale and zero parameters are set. "
                        "Setting zero=False.")
            params["zero"] = False

        # Parameters to be passed to stackSkyFrames
        stack_params = self._inherit_params(params, 'stackSkyFrames')
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
                log.warning(f"{ad.filename} has no SKYTABLE so cannot "
                            "subtract a sky frame")
                sky_list = None
            except KeyError:
                log.warning("Cannot read SKYTABLE associated with "
                            f"{ad.filename} so continuing")
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
                    log.warning(f"Cannot find a sky file named {filename}. "
                                "Ignoring it.")
                    skies.remove(filename)
                    continue
                else:
                    log.stdinfo(f"Found {filename} on disk")
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
                log.stdinfo(f"Cannot subtract sky from {ad.filename}")
                continue
            if stacked_skies[i] == 0:
                log.stdinfo(f"Creating sky frame for {ad.filename}")
                sky_inputs = [sky_dict[sky] for sky in skytable]
                stacked_sky = self.stackSkyFrames(sky_inputs, **stack_params)
                if len(stacked_sky) == 1:
                    stacked_sky = stacked_sky[0]
                    # Provide a more intelligent filename
                    if len(sky_inputs) > 1:
                        stacked_sky.filename = ad.filename
                        stacked_sky.update_filename(suffix="_sky", strip=True)
                else:
                    log.warning("Problem with stacking the following sky "
                                f"frames for {adinputs[i].filename}")
                    for filename in skytable:
                        log.warning(f"  {filename}")
                    stacked_sky = None
                # Assign this stacked sky frame to all adinputs that want it
                for j in range(i, len(skytables)):
                    if skytables[j] == skytable:
                        stacked_skies[j] = stacked_sky
                        if j > i:
                            log.stdinfo("This sky will also be used for "
                                        f"{adinputs[j].filename}")
                        skytables[j] = [None]

            # Go through all the science frames and sky-subtract any that
            # aren't needed for future sky-frame creation. We can get into
            # an issue here if we have two frames and are doing A-B and B-A
            # (plus other more complex examples) so we should sky-subtract
            # a frame even if it's a sky later on by making a copy.
            for j, ad2 in enumerate(adinputs):
                # If already been sky-subtracted or not yet processed
                if not skytables[j] or stacked_skies[j] == 0:
                    continue

                # We're iterating over *all* skytables so replace "None"s
                # with iterable empty lists
                frame_is_sky_for_future_skysub = ad2 in stacked_skies
                frame_needed_to_make_future_stacked_sky = (
                        ad2 in [sky_dict.get(sky) for skytable in skytables
                                for sky in (skytable or [])])
                if frame_is_sky_for_future_skysub:
                    adinputs[j] = self.subtractSky(
                        [deepcopy(ad2)], sky=stacked_skies[j], scale_sky=scale_sky,
                        offset_sky=offset_sky, reset_sky=reset_sky,
                        save_sky=save_sky, suffix=suffix).pop()
                elif not frame_needed_to_make_future_stacked_sky:
                    # Sky-subtraction is in-place, so we can discard the output
                    self.subtractSky([ad2], sky=stacked_skies[j], scale_sky=scale_sky,
                                     offset_sky=offset_sky, reset_sky=reset_sky,
                                     save_sky=save_sky, suffix=suffix)
                    skytables[j] = []
                    # This deletes a reference to the AD sky object
                    stacked_skies[j] = None

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
        debug_threshold = params["debug_threshold"]
        if scale and zero:
            log.warning("Both the scale_sky and offset_sky parameters are set. "
                        "Setting offset_sky=False.")
            zero = False

        # TODO: replace this by having min_sampled_pixels instead of sampling
        # in gt.measure_bg_from_image()
        sampling = 1 if adinputs[0].instrument() == 'GNIRS' else 10
        skyfunc = partial(gt.measure_bg_from_image, value_only=True,
                          sampling=sampling, gaussfit=True)

        for ad, ad_sky in zip(*gt.make_lists(adinputs, params["sky"],
                                             force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning(f"No changes will be made to {ad.filename}, since "
                            "it has already been processed by subtractSky")
                continue

            if ad_sky is not None:
                # Only call measure_bg_from_image if we need it
                if reset_sky:
                    orig_bg = skyfunc(ad)
                log.stdinfo(f"Subtracting {ad_sky.filename} from "
                            f"the science frame {ad.filename}")
                if scale or zero:
                    # This actually does the sky subtraction as well
                    try:
                        factors = [gt.sky_factor(ext, ext_sky, skyfunc,
                                                 multiplicative=scale,
                                                 threshold=debug_threshold)
                                   for ext, ext_sky in zip(ad, ad_sky)]
                    except ConvergenceError as error:
                        log.warning(f"The scaling of sky using a gaussian fit "
                                    f"did not converge.  \n"
                                    f"Using the median method instead.")
                        skyfunc = partial(gt.measure_bg_from_image,
                                          value_only=True,
                                          sampling=sampling, gaussfit=False)
                        try:
                            factors = [gt.sky_factor(ext, ext_sky, skyfunc,
                                                     multiplicative=scale)
                                       for ext, ext_sky in zip(ad, ad_sky)]
                        except ConvergenceError as error:
                            log.error(f"Failed to scaled sky.")
                            raise(error)

                    for ext_sky, factor in zip(ad_sky, factors):
                        log.fullinfo("Applying {} of {} to extension {}".
                                format("scaling" if scale else "offset",
                                       factor, ext_sky.id))
                else:
                    ad.subtract(ad_sky)
                if save_sky:
                    # Ensure the file written to disk has a filename that
                    # matches the frame from which it has been subtracted
                    ad_filename = ad.filename
                    ad.update_filename(suffix="_sky", strip=True)
                    log.stdinfo(f"Writing sky frame {ad.filename} to disk")
                    ad_sky.write(filename=ad.filename, overwrite=True)
                    ad.filename = ad_filename
                if reset_sky:
                    for ext, bg in zip(ad, orig_bg):
                        log.stdinfo(f"  Adding {bg} to {ad.filename} "
                                    f"extension {ext.id}")
                        ext.add(bg)
            else:
                log.warning(f"No changes will be made to {ad.filename}, "
                            "since no sky was specified")

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
                if bg is None:
                    log.warning(f"No changes will be made to {ad.filename} "
                                f"extension {ext.id}, since there "
                                "is no sky background measured")
                else:
                    log.fullinfo(f"Subtracting {bg:.0f} to remove sky level "
                                 f"from image {ad.filename} extension {ext.id}")
                    ext.subtract(bg)

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def thresholdFlatfield(self, adinputs=None, **params):
        """
        This primitive sets the DQ '64' bit (unilluminated) for any pixels
        which have a value <lower or >upper in the SCI plane.
        it also sets the science plane pixel value to 1.0 for pixels which are bad
        and very close to zero, to avoid divide by zero issues and inf values
        in the flat-fielded science data.

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
                unillum = np.where(((ext.data > upper) | (ext.data < lower)) &
                                   ((ext.mask & DQ.bad_pixel) == 0),
                                   DQ.unilluminated, 0).astype(DQ.datatype)
                ext.mask |= unillum
                log.fullinfo("ThresholdFlatfield set bit '64' for values "
                             "outside the range [{:.2f},{:.2f}]".
                             format(lower, upper))

                # Bad pixels might have low values and don't get flagged as
                # unilluminated, so we need to flag them to avoid infinite
                # values in the flat-fielded image
                ext.data[(ext.mask & DQ.unilluminated) > 0] = 1.0
                ext.data[ext.data < lower] = 1.0
                log.fullinfo("ThresholdFlatfield set flat-field pixels to 1.0 "
                             "for non-illuminated pixels.")

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)
        return adinputs

# -----------------------------------------------------------------------------
# Helper functions for scaleCountsToReference() follow
def mkcat_image(ad):
    """Produce a catalog of sources from a single-extension AstroData IMAGE"""
    cat = {}
    for ext in ad:
        try:
            objcat = ext.OBJCAT
            cat.update({(row['X_WORLD'], row['Y_WORLD']):
                        (row['FLUX_AUTO'], row['FLUXERR_AUTO']) for row in objcat})
        except (AttributeError, KeyError):
            pass
    if not cat:
        raise ValueError(f"{ad.filename} either has no OBJCAT(s) or they "
                        "are lacking the required columns")
    return cat


def calc_scaling_image(ad, ref_ad, objcat, ref_objcat, idx, matched, log):
    """Return an appropriate scaling"""
    ref_fluxes = np.array(list(ref_objcat.values()))[matched].T
    obj_fluxes = np.array(list(objcat.values()))[idx[matched]].T
    return at.calculate_scaling(x=obj_fluxes[0], y=ref_fluxes[0],
                                sigma_x=obj_fluxes[1], sigma_y=ref_fluxes[1])


def mkcat_spect(ad):
    """Produce a catalogue of sources from a spectroscopic AstroData object"""
    if len(ad[0].shape) == 1:  # extracted spectra
        cat = {(ra, dec): ad[i].nddata for i, (ra, dec) in
               enumerate(zip(ad.hdr['XTRACTRA'], ad.hdr['XTRACTDE']))}
    else:  # 2D spectrum with multiple sources
        try:
            aptable = ad[0].APERTURE
        except AttributeError:
            raise ValueError(f"{ad.filename} has no APERTURE table")
        dispaxis = 2 - ad.dispersion_axis()[0]  # python sense
        pix_coords = [0.5 * (length - 1) for length in ad[0].shape[::-1]]
        cat = {}
        for row in aptable:
            trace_model = am.table_to_model(row)
            aperture = tracing.Aperture(trace_model,
                                        aper_lower=row['aper_lower'],
                                        aper_upper=row['aper_upper'])
            pix_coords[dispaxis] = aperture.center
            wave, ra, dec = ad[0].wcs(*pix_coords)
            cat[ra, dec] = aperture
    return cat


def calc_scaling_spect(ad, ref_ad, objcat, ref_objcat, idx, matched, log):
    """Return an appropriate scaling"""
    ref_values = list(ref_objcat.values())
    values = list(objcat.values())
    data = None
    all_have_variance = True
    size = 0
    for i, (j, m) in enumerate(zip(idx, matched)):
        if m:
            log.stdinfo(f"Matched aperture {i+1} from {ref_ad.filename} "
                        f"with aperture {j+1} from {ad.filename}")
            ref_ap = ref_values[i]
            if isinstance(ref_ap, tracing.Aperture):
                log.debug(f"Extracting aperture {i+1} from {ref_ad.filename}")
                ref_ap = ref_ap.extract(ref_ad[0])
                ref_objcat[list(ref_objcat.keys())[i]] = ref_ap
            ap = values[j]
            if isinstance(ap, tracing.Aperture):
                log.debug(f"Extracting aperture {j+1} from {ad.filename}")
                ap = ap.extract(ad[0])
                objcat[list(objcat.keys())[j]] = ap
            if data is None:
                data = np.empty((4, matched.sum() * ap.shape[0]))
            good = np.ones_like(ref_ap.data, dtype=bool)
            if ap.mask is not None:
                good[:] &= ap.mask == 0
            if ref_ap.mask is not None:
                good[:] &= ref_ap.mask == 0
            if ap.variance is not None:
                good[:] &= ap.variance > 0
            if ref_ap.variance is not None:
                good[:] &= ref_ap.variance > 0
            ngood = good.sum()
            data[:2, size:size+ngood] = np.array([ref_ap.data[good],
                                                  ap.data[good]])
            if ap.variance is None or ref_ap.variance is None:
                all_have_variance = False
            else:
                data[2:, size:size+ngood] = np.array([ref_ap.variance[good],
                                                      ap.variance[good]])
            size += ngood
    if size > 0:
        if all_have_variance:
            return at.calculate_scaling(x=data[1, :size], y=data[0, :size],
                                        sigma_x=np.sqrt(data[3, :size]),
                                        sigma_y=np.sqrt(data[2, :size]))
        return at.calculate_scaling(x=data[1, :size], y=data[0, :size])
    return -1
