#
#
#
#                                                  primitives_crossdispersed.py
# -----------------------------------------------------------------------------

from abc import abstractmethod
from copy import deepcopy
from importlib import import_module

import astrodata, gemini_instruments
from astrodata.utils import Section

from astropy.modeling import models
from astropy.table import Table, vstack
from gwcs import coordinate_frames as cf
from gwcs.wcs import WCS as gWCS
import numpy as np
from recipe_system.utils.decorators import (parameter_override,
                                            capture_provenance)

from gempy.gemini import gemini_tools as gt
from gempy.library import astromodels as am
from geminidr.core import Spect, Preprocess
from . import parameters_crossdispersed


@parameter_override
@capture_provenance
class CrossDispersed(Spect, Preprocess):
    """This is the class containing primitives specifically for crossdispersed
    data. It inherits all the primitives from the level above.

    """
    tagset = {'GEMINI', 'SPECT', 'XD'}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_crossdispersed)

    def cutSlits(self, adinputs=None, **params):
        """
        Extract slits in images into individual extensions.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Data as 2D spectral images with slits defined in a SLITEDGE table.
        suffix :  str
            Suffix to be added to output files.

        Returns
        -------
        list of :class:`~astrodata.AstroData`

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]

        order_key_parts = self._get_order_information_key()
        order_info = import_module('.orders_XD', self.inst_lookups).order_info

        adoutputs = []
        for ad in adinputs:
            order_key = "_".join(getattr(ad, desc)() for desc in order_key_parts)
            _, dispersions = order_info[order_key]

            # Get the central wavelength setting and order it occurs in.
            central_wavelength = ad.central_wavelength(asNanometers=True)
            grating_order = ad._grating_order()

            # This is the presumed pointing location and the centres of
            # each cut slit should recover these sky coordinates
            world_refpos = ad[0].wcs(*list(0.5 * (length - 1)
                                           for length in ad[0].shape[::-1]))
            ad = self._cut_slits(ad, padding=2)

            for i, ext in enumerate(ad):
                dispaxis = 2 - ext.dispersion_axis()  # Python Sense
                specaxis_middle = 0.5 * (ext.shape[dispaxis] - 1)
                try:
                    slit_center = np.mean([am.table_to_model(row)(specaxis_middle)
                                           for row in ext.SLITEDGE])
                except AttributeError:
                    continue

                # Get the spectral order for this extension
                try:
                    spec_order = set(ext.SLITEDGE["specorder"])
                except KeyError:
                    if 'XD' in ad.tags:
                        raise RuntimeError("No order information found in "
                                           f"SLITEDGE for {ad.filename}")
                else:
                    if len(spec_order) > 1:
                        raise RuntimeError("Multiple orders found in SLITEDGE")
                    spec_order = spec_order.pop()
                    ext.hdr['SPECORDR'] = spec_order

                # Update the central wavelength for this order using the
                # following formula:
                #   order_X * cent_wavelength_X = order_Y * cent_wavelength_Y
                # e.g., 3 * 2.3 = 4 * central_wavelength_4 -> 3/4 * 2.2 = 1.65
                # We have the central wavelength and number of one order from
                # the header, so we can find the central wavelength in any
                # other order.
                centwl = grating_order * central_wavelength / spec_order

                # Update the WCS by adding a "first guess" wavelength scale
                # for each slit.
                new_wave_model = (models.Shift(-specaxis_middle) |
                                  models.Scale(dispersions[i]) |
                                  models.Shift(centwl))
                new_wave_model.name = "WAVE"

                for idx, step in enumerate(ext.wcs.pipeline):
                    try:
                        ext.wcs.pipeline[idx] = step.__class__(
                            step.frame, step.transform.replace_submodel(
                                "WAVE", new_wave_model))
                    except (AttributeError, ValueError):  # not in this step
                        pass
                    else:
                        # Update the SKY model so all slit centers point to
                        # the same location
                        coords = ext.wcs.invert(centwl, *world_refpos[1:])
                        shift = coords[dispaxis] - slit_center
                        sky_model = am.get_named_submodel(step.transform, "SKY")
                        new_sky_model = models.Shift(shift) | sky_model
                        # "step" hasn't been updated with the new WAVE model
                        ext.wcs.pipeline[idx] = step.__class__(
                            step.frame, ext.wcs.pipeline[idx].transform.replace_submodel(
                                "SKY", new_sky_model))
                        sky_model.name = None
                        new_sky_model.name = "SKY"
                        break
                else:
                    log.warning("No initial wavelength model found - "
                                "not updating the wavelength model")



            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

            adoutputs.append(ad)

        return adoutputs

    def _cut_slits(self, ad, padding=0):
        # TODO: This should move to Spect but it's left here for now to
        # ease code maintenance since it's only used by XD
        """
        This method cuts a 2D spectral image into individual slits based on
        an existing SLITEDGE table. The slits are then stored in individual
        extensions. A Chebyshev2D model is created to represent the slit
        distortion and this is inserted into the WCS. No work is performed
        to modify the astrometry since this is the responsibility of the
        calling primitive and will depend on the type of data (XD/MOS/IFU).

        Parameters
        ----------
        ad: AstroData object
            2D spectral image with slits defined in a SLITEDGE table
        padding: int
            additional padding to put beyond edges of slits when cutting
            (to account for possible misalignments)

        Returns
        -------
        modified AstroData object with multiple extensions
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        adout = astrodata.create(ad.phu)
        adout.filename = ad.filename
        adout.orig_filename = ad.orig_filename
        detsec_kw = ad._keyword_for('detector_section')
        binnings = ad.detector_x_bin(), ad.detector_y_bin()

        for ext in ad:
            try:
                slitedge = ext.SLITEDGE
            except AttributeError:
                log.warning(f"No slits to cut in {ad.filename}:{ext.id}")
                adout.append(ext)
                continue

            orig_wcs = ext.wcs
            dispaxis = 2 - ext.dispersion_axis()  # Python Sense

            # Iterate over pairs of edges in the SLITEDGE table
            # In the following code, "x"  represents the location along the
            # spectral axis, and "y" the orthogonal direction (which is a
            # function of x). This is opposite to F2 and GNIRS.
            for i, (slit1, slit2) in enumerate(zip(_ := iter(slitedge), _)):
                model1 = am.table_to_model(slit1)
                model2 = am.table_to_model(slit2)

                # Reconstruct the slit edges from the models and identify
                # which parts are on the detector
                xpixels = np.arange(ext.shape[dispaxis])
                ypixels1, ypixels2 = model1(xpixels), model2(xpixels)
                ypix1_on = np.logical_and(
                    ypixels1 >= padding, ypixels1 <= ext.shape[1 - dispaxis] - padding - 1)
                ypix2_on = np.logical_and(
                    ypixels2 >= padding, ypixels2 <= ext.shape[1 - dispaxis] - padding - 1)
                y1 = int(max(np.floor(ypixels1.min() - padding), 0))
                y2 = int(min(np.ceil(ypixels2.max() + padding),
                             ext.shape[1 - dispaxis]))

                # Add cut rectangle and a modified SLITEDGE table with
                # the single slit for this extension
                if dispaxis == 0:
                    cut_section = Section(x1=y1, x2=y2, y1=0, y2=ext.shape[0])
                else:
                    cut_section = Section(x1=0, x2=ext.shape[1], y1=y1, y2=y2)
                log.stdinfo(f"Cutting slit {i+1} in extension {ext.id} "
                            f"from {cut_section.asIRAFsection()}")
                adout.append(deepcopy(ext.nddata[cut_section.asslice()]))
                adout[-1].SLITEDGE = slitedge[i*2:i*2+2]
                adout[-1].SLITEDGE["c0"] -= y1
                adout[-1].SLITEDGE["slit"] = 1  # reset slit number in ext

                # Calculate a Chebyshev2D model that represents both slit
                # edges. This requires coordinates be fed with the *detector*
                # x-coordinate first. The rectified slit will be as wide in
                # pixels as it is halfway up the unrectified image, and the
                # left edge will be "padding" pixels in from the left edge of
                # the image.
                xcenter = 0.5 * (ext.shape[dispaxis] - 1)
                y1ref = np.full_like(ypixels1, padding)[ypix1_on]
                y2ref = np.full_like(ypixels2, model2(xcenter) - model1(xcenter) + padding)[ypix2_on]
                log.debug(f"Slit at {xcenter} from "
                          f"{y1ref[0] if len(y1ref) else 'edge'} to "
                          f"{y2ref[0] if len(y2ref) else 'edge'}")

                if dispaxis == 0:
                    incoords = [np.r_[ypixels1[ypix1_on] - y1, ypixels2[ypix2_on] - y1],
                                np.r_[xpixels[ypix1_on], xpixels[ypix2_on]]]
                    refcoords = [np.r_[y1ref, y2ref], np.r_[xpixels[ypix1_on], xpixels[ypix2_on]]]
                    xorder, yorder = 1, model1.degree
                else:
                    incoords = [np.r_[xpixels[ypix1_on], xpixels[ypix2_on]],
                                np.r_[ypixels1[ypix1_on] - y1, ypixels2[ypix2_on] - y1]]
                    refcoords = [np.r_[xpixels[ypix1_on], xpixels[ypix2_on]], np.r_[y1ref, y2ref]]
                    xorder, yorder = model1.degree, 1

                m_init_2d = models.Chebyshev2D(
                    x_degree=xorder, y_degree=yorder,
                    x_domain=[0, ext.shape[1]-1],
                    y_domain=[0, ext.shape[0]-1])
                log.stdinfo("  Creating distortion model for slit "
                            f"rectification for slit {i+1}")
                # The `fixed_linear` parameter is False because we should
                # have both edges for each slit.
                model, m_final_2d, m_inverse_2d = am.create_distortion_model(
                    m_init_2d, dispaxis, incoords, refcoords, fixed_linear=False)
                model.name = "RECT"

                # Remove the shift that was prepended when the data were
                # sliced -- the easiest way to do this is just to use the
                # original WCS (this will mess up the astrometry)
                adout[-1].wcs = deepcopy(orig_wcs)
                if adout[-1].wcs is None:
                    adout[-1].wcs = gWCS([(ext.wcs.input_frame, model),
                                          (cf.Frame2D(name="rectified"),
                                           None)])
                else:
                    adout[-1].wcs.insert_frame(ext.wcs.input_frame, model,
                                               cf.Frame2D(name="rectified"))

                # TODO: this updates the detector_section keyword, which
                # is fine for instruments with only one array in the
                # detector. NEEDS UPDATING for multi-array instruments
                # (e.g., MOS data from MOS), and will need to update
                # the array_section keyword then as well.
                adout[-1].hdr[detsec_kw] = (
                    cut_section.asIRAFsection(binning=binnings),
                    self.keyword_comments.get(detsec_kw))

        return adout

    def findApertures(self, adinputs=None, **params):
        """
        Finds sources in 2D spectral images and store them in an APERTURE table
        for each extension. Each table will, then, be used in later primitives
        to perform aperture extraction.

        The primitive operates by first collapsing the 2D spectral image in
        the spatial direction to identify sky lines as regions of high
        pixel-to-pixel variance, and the regions between the sky lines which
        consist of at least `min_sky_region` pixels are selected. These are
        then collapsed in the dispersion direction to produce a 1D spatial
        profile, from which sources are identified using a peak-finding
        algorithm.

        The widths of the apertures are determined by calculating a threshold
        level relative to the peak, or an integrated flux relative to the total
        between the minima on either side and determining where a smoothed
        version of the source profile reaches this threshold.

        This version handles concerns related to cross-dispersed images. It
        will create a new AstroData object with a single extension: either one
        provided by the user, or the "best" (as in brightest) extension if the
        user does not provide one

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Science data as 2D spectral images.
        suffix : str
            Suffix to be added to output files.
        max_apertures : int
            Maximum number of apertures expected to be found.
        percentile : float (0 - 100) / None
            percentile to use when collapsing along the dispersion direction
            to obtain a slit profile / None => take mean.
        section : str
            comma-separated list of colon-separated pixel coordinate pairs
            indicating the region(s) over which the spectral signal should be
            used. The first and last values can be blank, indicating to
            continue to the end of the data.
        min_sky_region : int
            minimum number of contiguous pixels between sky lines
            for a region to be added to the spectrum before collapsing to 1D.
        min_snr : float
            minimum S/N ratio for detecting peaks.
        use_snr : bool
            Convert data to SNR per pixel before collapsing and peak-finding?
        threshold : float (0 - 1)
            parameter describing either the height above background (relative
            to peak) at which to define the edges of the aperture.
        ext : int
            The number of the extension (1-indexed) to look for apertures in.
            If not given, an extension will be chosen based on the value of
            `comp_method`.
        comp_method : str, Default : "sum"
            Comparison method to use for determining "best" order for finding
            apertures in.
        interactive : bool
            Show interactive controls for fine tuning source aperture detection.

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            The 2D spectral images with APERTURE tables attached

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        interactive = params["interactive"]
        ext = params.pop("ext")
        comp_method = params.pop("comp_method")

        # Need to send findApertures a new AstroData object with one extension,
        # containing either the user-specified or "best" order.
        for ad in adinputs:
            dispaxis = 2 - ad[0].dispersion_axis()  # Python Sense

            if ext is None:
                # We need to find the "best" order to perform the trace in. For
                # now, this is (somewhat simplistically) determined as the order
                # with the most flux, or the highest median flux value. This
                # could fail if the image were overexposed, but it works well
                # enough as a first approximation.
                discriminant = []
                for ext in ad:
                    data = ext.data[ext.mask == 0].ravel()
                    discriminant.append(getattr(np, comp_method)(data))
                ext_num = np.array(discriminant).argmax()
            else:
                ext_num = ext - 1 # Change to 0-indexed

            # Create a temporary AstroData object.
            temp_ad = astrodata.create(ad.phu)
            temp_ad.filename = "temporary file"
            temp_ad.append(ad[ext_num])

            log.stdinfo(f"Looking in extension {ext_num+1} of {ad.filename}")

            # Perform aperture-finding in new AD object with one extension
            ap_found = super().findApertures(adinputs=[temp_ad], **params).pop()
            try:
                ap_table = ap_found[0].APERTURE
            except AttributeError:
                # No apertures found, just continue (warning handled by
                # Spect.findApertures)
                continue
            source_pos = ap_table['c0']

            # Now need to translate the pixel coordinates found for the aperture
            # to world -> back to pixels for each other extension, since the
            # extensions may have different dimensions.
            for ext in ad:
                # The upper and lower edges should be the same in each order
                # (since they're relative to the center and the pixel scale
                # is the same in each order), so copy the whole APERTURE table
                # and just update the center position(s) in each extension.
                ext.APERTURE = deepcopy(ap_table)

                for i, source in enumerate(source_pos):
                    if dispaxis == 0:
                        x, y = source, ap_found[0].shape[0] / 2
                    else:
                        x, y = ap_found[0].shape[1] / 2, source

                    new_source_pos = ext.wcs.backward_transform(
                        *ap_found[0].wcs.forward_transform(x, y))[0 - dispaxis]
                    log.fullinfo(f"Copying aperture {i} to extension {ext.id} "
                                 f"at position {new_source_pos}.")

                    ext.APERTURE['c0'][i] = new_source_pos

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def flatCorrect(self, adinputs=None, **params):
        """
        This primitive will divide each SCI extension of the inputs by those
        of the corresponding flat. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the division on the data.
        If no flatfield is provided, the calibration database(s) will be
        queried.

        If the flatfield has had a QE correction applied, this information is
        copied into the science header to avoid the correction being applied
        twice.

        This primitive calls the version of flatCorrect in primitives_preprocess
        after first cutting the data into multiple extensions to match the flat.
        Arguments will be passed up to flatCorrect via a call to super().

        Parameters
        ----------
        suffix: str
            Suffix to be added to output files.
        flat: str
            Name of flatfield to use.
        do_flat: bool
            Perform flatfield correction?

        Returns
        -------
        list of :class:`~astrodata.AstroData`
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        flat = params.pop('flat')
        do_cal = params['do_cal']

        if flat is None:
            flat_list = self.caldb.get_processed_flat(adinputs)
        else:
            flat_list = (flat, None)

        for ad, flat, origin in zip(*gt.make_lists(adinputs, *flat_list,
                                    force_ad=(1,))):
            if flat is None:
                if 'sq' in self.mode or do_cal == 'force':
                   raise OSError("No processed flat listed for "
                                 f"{ad.filename}")
                else:
                   log.warning(f"No changes will be made to {ad.filename}, "
                               "since no flatfield has been specified")
                   continue

            if len(ad) != 1:
                log.warning(f"{ad.filename} has more than one extension and "
                            "will not be flatfielded")
                continue

            # CJS: For speed of getting something working, we're going to
            # reconstruct the original SLITEDGE model from the flatfield
            # and then use it to cut the science data. So, rather than copy
            # the rectification model from the flat, a completely new one
            # will be constructed, but it will be identical to that one in
            # the flat, since it is calculated from the same SLITEDGE table.
            # (although, as noted below, the one from the flat gets copied).
            # cutSlits() will also sort out the WCS for each cut extension.
            slitedge = vstack([flat_ext.SLITEDGE for flat_ext in flat],
                              metadata_conflicts='silent')
            for i, (flat_detsec, dispaxis) in enumerate(zip(flat.detector_section(),
                                                            flat.dispersion_axis())):
                offset = flat_detsec.x1 if dispaxis == 2 else flat_detsec.y1
                slitedge[i*2:i*2+2]["c0"] += offset
            ad[0].SLITEDGE = slitedge

        adinputs = self.cutSlits(adinputs, suffix=None)

        # Since we've already worked out the flats to use, send them along to
        # avoid needing to re-query the Caldb
        try:
            flat_files = flat_list.files
        except AttributeError:  # not a CalReturn object
            flat_files = flat_list[0]

        # flatCorrect() will actually copy the rectification model from the
        # flat, overwriting the identical model we created above.
        # This will timestamp and update the suffix for us
        adinputs = super().flatCorrect(adinputs, flat=flat_files, **params)

        return adinputs

    def _make_tab_labels(self, ad):
        """
        Create tab labels for cross-dispersed data.

        Parameters
        ----------
        ad : `~astrodata.AstroData`
            The AstroData object to be processed.

        Returns
        -------
        list
            A list of tab labels for the given AstroData object.
        """
        apertures = ad.hdr.get('APERTURE')
        orders = ad.hdr.get('SPECORDR')
        num_ap = len(set(apertures))
        tab_labels = []
        for ap, ord in zip(apertures, orders):
            label = f"Aperture {ap} " if num_ap > 1 else ""
            label += f"Order {ord}"
            tab_labels.append(label)
        return tab_labels
