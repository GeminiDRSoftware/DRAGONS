#
#                                                                 gemini_python
#
#                                                     primitives_f2_longslit.py
# -----------------------------------------------------------------------------

from astropy.table import Table
import numpy as np

from gempy.gemini import gemini_tools as gt
from recipe_system.utils.decorators import (parameter_override,
                                            capture_provenance)
from geminidr.gemini.lookups import DQ_definitions as DQ
from gemini_instruments.f2.lookup import dispersion_offset_mask

from . import parameters_f2_longslit
from .primitives_f2_spect import F2Spect

# -----------------------------------------------------------------------------

# This dictionary contains the following information for each slit configuration:
# x-coordinate of the pixel in the center of the slit, slit length in arcsec,
# slit length in pixels. The F2 pixel scale is 0.18 arsec/pixel, according to
# the instrument webpage.
f2_slit_info = {'1pix': (763.5, 271.62, 1509),
                '2pix': (769.5, 265.14, 1473),
                '3pix': (770.0, 271.80, 1510),
                '4pix': (772.5, 270.54, 1503),
                '6pix': (777.0, 271.80, 1510),
                '8pix': (771.0, 271.80, 1510)}

@parameter_override
@capture_provenance
class F2Longslit(F2Spect):
    """This class contains all of the processing primitives for the F2Longslit
    level of the type hiearchy tree. It inherits all the primitives from the
    above level.
    """
    tagset = {'GEMINI', 'F2', 'SPECT', 'LS'}
    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_f2_longslit)

    def _fields_overlap(self, ad1, ad2, frac_FOV=1.0, max_perpendicular_offset=None):
        slit_length = 1300 * ad1.pixel_scale() if ad1.is_ao()\
                                            else ad1.MDF['slitlength_arcsec'][0]
        slit_width = int(ad1.focal_plane_mask()[0]) * ad1.pixel_scale()
        return super()._fields_overlap(
            ad1, ad2, frac_FOV=frac_FOV, slit_length=slit_length,
            slit_width=slit_width, max_perpendicular_offset=max_perpendicular_offset)

    def addMDF(self, adinputs=None, suffix=None, mdf=None):
        """
        This F2-specific implementation of addMDF() adds a "virtual MDF" (as
        in, created from data in this module rather than pulled from another
        file) to F2 data. It calls primitives_gemini._addMDF() on each astrodata
        object to attach the MDFs, then performs corrections depending on the
        data. It also attaches two columns, 'slitsize_arcsec' and 'slitsize_pixels'
        with the length of the slit in arcseconds and pixels, respectively.

        Parameters
        ----------
        suffix : str
            suffix to be added to output files
        mdf : str/None
            name of MDF to add (None => use default)

        """

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        mdf_list = mdf or self.caldb.get_calibrations(adinputs,
                                                      caltype="mask").files

        for ad, mdf in zip(*gt.make_lists(adinputs, mdf_list, force_ad=True)):

            # F2 doesn't have actual mask defintiion files, so this won't add
            # anyything, but it will check if the file already has an MDF table
            self._addMDF(ad, suffix, mdf)

            if hasattr(ad, 'MDF'):
                continue
            else:
                log.stdinfo(f"Creating MDF table for F2 file {ad.filename}.")

                maskname = ad.focal_plane_mask(pretty=True).split('-')[0]

                mdf_table = Table(np.array(f2_slit_info[maskname]),
                                  names=('x_ccd', 'slitlength_arcsec',
                                         'slitlength_pixels'))
                ad.MDF = mdf_table

        return adinputs

    def addIllumMaskToDQ(self, adinputs=None, suffix=None, illum_mask=None):
        """
        Adds an illumination mask to each AD object. The default illumination mask
        masks off extra orders and/or unilluminated areas outside order blocking filter
        range (whenever required).

        Parameters
        ----------
        suffix : str
            suffix to be added to output files
        illum_mask : str/None
            name of illumination mask mask (None -> use default)

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        for ad, illum in zip(*gt.make_lists(adinputs, illum_mask, force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning('No changes will be made to {}, since it has '
                    'already been processed by addIllumMaskToDQ'.
                            format(ad.filename))
                continue
            if illum:
                log.fullinfo("Using {} as illumination mask".format(illum.filename))
                final_illum = gt.clip_auxiliary_data(ad, aux=illum, aux_type='bpm',
                                          return_dtype=DQ.datatype)

                for ext, illum_ext in zip(ad, final_illum):
                    if illum_ext is not None:
                        # Ensure we're only adding the unilluminated bit
                        iext = np.where(illum_ext.data > 0, DQ.unilluminated,
                                        0).astype(DQ.datatype)
                        ext.mask |= iext

            else:
                dispaxis = 2 - ad[0].dispersion_axis()
                dispaxis_center = ad[0].shape[dispaxis] // 2
                cenwave = ad.central_wavelength(asNanometers=True)
                dispersion = ad.dispersion(asNanometers=True)[0]
                filter = ad.filter_name(pretty=True)
                if filter in {"HK", "JK"}:
                        filter = ad.filter_name(keepID=True)
                cenwave_offset = self._get_cenwave_offset(ad)
                index = (ad.disperser(pretty=True), filter)
                mask = dispersion_offset_mask.get(index, None)

                filter_cuton_wvl = mask.cutonwvl if mask else None
                filter_cutoff_wvl = mask.cutoffwvl if mask else None
                cenwave_pix = dispaxis_center + cenwave_offset
                filter_cuton_pix = min(int(cenwave_pix - (cenwave - filter_cuton_wvl) / dispersion), ad[0].shape[dispaxis] - 1)
                filter_cutoff_pix = max(int(cenwave_pix + (filter_cutoff_wvl - cenwave) / dispersion), 0)

                for ext in ad:
                    ext.mask[:filter_cutoff_pix] |= DQ.unilluminated
                    ext.mask[filter_cuton_pix:] |= DQ.unilluminated
                if filter_cutoff_pix > 0:
                    log.stdinfo(f"Masking columns 1 to {filter_cutoff_pix+1}")
                if filter_cuton_pix < (ad[0].shape[dispaxis] - 1):
                    log.stdinfo(f"Masking columns {filter_cuton_pix+1} to {(ad[0].shape[dispaxis])}")

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs
