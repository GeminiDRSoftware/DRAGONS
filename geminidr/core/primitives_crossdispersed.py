#
#
#
#                                                  primitives_crossdispersed.py
# -----------------------------------------------------------------------------

from abc import abstractmethod
from copy import deepcopy
from importlib import import_module

import astrodata, gemini_instruments
from astropy.modeling import models
from astropy.table import Table
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

    @abstractmethod
    def _map_spec_order(self, ext_id):
        pass

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

        adinputs = super()._cut_slits(adinputs=adinputs, **params)

        columns = ('central_wavelength', 'dispersion', 'center_offset')
        order_key_parts = self._get_order_information_key()

        adoutputs = []
        for ad in adinputs:
            order_key = "_".join(getattr(ad, desc)() for desc in order_key_parts)
            # order_info contains central wavelength, dispersion, and offset
            # (in pixels) for each slit.
            order_info = Table(import_module(
                '.orders_XD_GNIRS', self.inst_lookups).order_info[order_key],
                names=columns)

            for i, ext in enumerate(ad):
                dispaxis = 2 - ext.dispersion_axis()  # Python Sense

                # Update the WCS by adding a "first guess" wavelength scale
                # for each slit.
                for idx, step in enumerate(ext.wcs.pipeline):
                    if ext.wcs.pipeline[idx+1].frame.name == 'world':
                        # Need to find a sequence of Shift + Scale models, which
                        # collectively represent the wavecal model.
                        _, m_wave_sky = step
                        found_wavecal_model, found_sky_model = False, False
                        for j in range(m_wave_sky.n_submodels):
                            if (isinstance(m_wave_sky[j], models.Shift) and
                                isinstance(m_wave_sky[j+1], models.Scale) and
                                isinstance(m_wave_sky[j+2], models.Shift)):

                                found_wavecal_model = True
                                break

                        if found_wavecal_model:
                            # The first shift is the offset to the center of the
                            # array, don't modify it.
                            # Dispersion (nm/pixel)
                            step.transform[j+1].factor = order_info[i]['dispersion']
                            # Central wavelength offset (nm)
                            step.transform[j+2].offset = order_info[i]['central_wavelength']

                            log.fullinfo(f"Updated WCS for extension {ext.id}")
                            break
                        else:
                            log.warning("No initial wavelength model found - "
                                        "not modifying the WCS")
                        break

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

            adoutputs.append(ad)

        return adoutputs

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
            ap_table = ap_found[0].APERTURE
            source_pos = ap_table['c0']

            # Now need to translate the pixel coordinates found for the aperture
            # to world -> back to pixels for each other extension, since the
            # extensions may have different dimensions.
            for iext, ext in enumerate(ad):
                # The upper and lower edges should be the same in each order
                # (since they're relative to the center and the pixel scale
                # is the same in each order), so copy the whole APERTURE table
                # and just update the center position(s) in each extension.
                ext.APERTURE = deepcopy(ap_table)

                for i, source in enumerate(source_pos):
                    log.debug(f"Copying source {i} to order {iext}.")
                    if dispaxis == 0:
                        x, y = source, ap_found[0].shape[0] / 2
                    else:
                        x, y = ap_found[0].shape[1] / 2, source

                    new_source_pos = ext.wcs.backward_transform(
                        *ap_found[0].wcs.forward_transform(x, y))[0 - dispaxis]

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
        timestamp_key = self.timestamp_keys[self.myself()]
        flat = params['flat']
        do_flat = params.get('do_flat', None)
        sfx = params["suffix"]

        if flat is None:
            flat_list = self.caldb.get_processed_flat(adinputs)
        else:
            flat_list = (flat, None)

        cut_ads = []
        for ad, flat, origin in zip(*gt.make_lists(adinputs, *flat_list,
                                    force_ad=(1,))):
            if flat is None:
                if 'sq' in self.mode or do_flat == 'force':
                   raise OSError("No processed flat listed for "
                                 f"{ad.filename}")
                else:
                   log.warning(f"No changes will be made to {ad.filename}, "
                               "since no flatfield has been specified")
                   continue

            # Cut the science frames to match the flats, which have already
            # been cut in the flat reduction.
            cut_ad = gt.cut_to_match_auxiliary_data(adinput=ad, aux=flat)
            # Need to update the WCS in each extension by replacing its WAVE
            # model with the one from the corresponding extension in the flat.
            # To do that, recreate the WCS with the original input and output
            # frames and the new wavecal mode taken from the flat.
            for ext, auxext in zip(cut_ad, flat):
                aux_wave_model = am.get_named_submodel(auxext.wcs.forward_transform,
                                                       'WAVE')
                new_wave_model = ext.wcs.forward_transform.replace_submodel(
                    "WAVE", aux_wave_model)
                ext.wcs = gWCS([(ext.wcs.input_frame, new_wave_model),
                                (ext.wcs.output_frame, None)])
                log.fullinfo(f'WCS wavecal model updated in ext {ext.id}')

            cut_ads.append(cut_ad)

        adoutputs = super().flatCorrect(adinputs=cut_ads, **params)
        for ad in adoutputs:
            try:
                ad[0].wcs.get_transform("pixels", "rectified")
            except:
                log.warning(f"No rectification model found for {ad.filename}")

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adoutputs
