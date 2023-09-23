#
#
#
#                                                  primitives_crossdispersed.py
# -----------------------------------------------------------------------------

from importlib import import_module

from astropy.modeling import models
from astropy.table import Table
from recipe_system.utils.decorators import (parameter_override,
                                            capture_provenance)

from gempy.gemini import gemini_tools as gt
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

            for j, ext in enumerate(ad):
                dispaxis = 2 - ext.dispersion_axis()  # Python Sense
                row = order_info[j]

                # Handle the WCS. Need to adjust it for each slit.
                for idx, step in enumerate(ext.wcs.pipeline):
                    if ext.wcs.pipeline[idx+1].frame.name == 'world':
                        if not (isinstance(step.transform[4], models.Scale)
                            and isinstance(step.transform[5], models.Shift)):
                            log.warning("No initial wavelength model found - "
                                        "not modifying the WCS")
                            break

                        # Central wavelength offset (nm)
                        step.transform[5].offset = row['central_wavelength']
                        # Dispersion (nm/pixel)
                        step.transform[4].factor = row['dispersion']
                        # Offset of center of slit (pixels) - The order of sub-
                        # models in the WCS transform is independent of
                        # dispersion axis, so the Shift model
                        # index changes depending on orientation.
                        crpix_index = 7 if dispaxis == 0 else 3
                        step.transform[crpix_index].offset = row['center_offset']
                        log.fullinfo(f"Updated WCS for ext {ext.id}")
                        break

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

            adoutputs.append(ad)

        return adoutputs

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
            cut_ads.append(gt.cut_to_match_auxiliary_data(adinput=ad, aux=flat))

            # Copy the WCS from each extension in the flat to the corresponding
            # extension in the file:
            for i in range(len(cut_ads[-1])):
                init_wave_model = flat[i].wcs.get_transform('rectified',
                                                            'world')
                # The AD being corrected doesn't yet have a 'rectified' frame,
                # just 'pixels' and 'world', so insert this transform between them.
                cut_ads[-1][i].wcs.set_transform('pixels', 'world',
                                                 init_wave_model)

        adoutputs = super().flatCorrect(adinputs=cut_ads, **params)

        return adoutputs
