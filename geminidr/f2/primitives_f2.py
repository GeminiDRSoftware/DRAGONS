#
#                                                                  gemini_python
#
#                                                               primitives_f2.py
# ------------------------------------------------------------------------------
import numpy as np

from gempy.gemini import gemini_tools as gt
from gempy.adlibrary.manipulate_ad import remove_single_length_dimension

from geminidr.core import NearIR
from geminidr.gemini.primitives_gemini import Gemini
from . import parameters_f2
from gemini_instruments.f2 import lookup as adlookup

from recipe_system.utils.decorators import parameter_override, capture_provenance


@parameter_override
@capture_provenance
class F2(Gemini, NearIR):
    """
    This is the class containing all of the preprocessing primitives
    for the F2 level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = {"GEMINI", "F2"}

    def _initialize(self, adinputs, **kwargs):
        self.inst_lookups = 'geminidr.f2.lookups'
        self.inst_adlookup = adlookup
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_f2)

    def standardizeInstrumentHeaders(self, adinputs=None, **params):
        """
        This primitive is used to make the changes and additions to the
        keywords in the headers of F2 data, specifically.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        log.status("Updating keywords that are specific to FLAMINGOS-2")
        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning(f"No changes will be made to {ad.filename}, "
                            "since it has already been processed by "
                            "standardizeInstrumentHeaders")
                continue

            # Standardize the headers of the input AstroData object. Update the
            # keywords in the headers that are specific to FLAMINGOS-2.

            # Filter name (required for IRAF?)
            ad.phu.set('FILTER', ad.filter_name(stripID=True, pretty=True),
                       self.keyword_comments['FILTER'])

            # Pixel scale (CJS: I'm putting this in the extension too!)
            pixel_scale = ad.pixel_scale()
            ad.phu.set('PIXSCALE', pixel_scale, self.keyword_comments['PIXSCALE'])
            ad.hdr.set('PIXSCALE', pixel_scale, self.keyword_comments['PIXSCALE'])

            # KL: fix the WCS for AO
            if ad.is_ao():
                for ext in ad:
                    ext.wcs.forward_transform['cd_matrix'].matrix *= \
                        ext.pixel_scale() / ext._get_wcs_pixel_scale()

            for desc in ('read_noise', 'gain', 'non_linear_level',
                         'saturation_level', 'array_section',
                         'data_section', 'detector_section'):
                kw = ad._keyword_for(desc)
                desc_params = {"pretty": True} if 'section' in desc else {}
                ad.hdr.set(kw, getattr(ad, desc)(**desc_params)[0], self.keyword_comments[kw])
                try:
                    ad.phu.remove(kw)
                except (KeyError, AttributeError):
                    pass

            if 'SPECT' in ad.tags:
                kw = ad._keyword_for('dispersion_axis')
                ad.hdr.set(kw, 2, self.keyword_comments[kw])

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
            log.debug(f"Successfully updated keywords for {ad.filename}")
        return adinputs

    def standardizeStructure(self, adinputs=None, **params):
        """
        This primitive is used to standardize the structure of F2 data,
        specifically.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        attach_mdf: bool
            attach an MDF to the AD objects? (ignored if not tagged as SPECT)
        mdf: str
            full path of the MDF to attach
        """
        adinputs = super().standardizeStructure(adinputs, **params)

        # Raw FLAMINGOS-2 pixel data have three dimensions (2048x2048x1).
        # Remove the single length dimension from the pixel data.
        # CD3_3 keyword must also be removed.
        # Also, F2 data are written with BITPIX=32, which astropy.io.fits
        # turns to np.float64. But the raw data values could be represented
        # as 16-bit integers, so we downgrade to np.float32 here
        for ad in adinputs:
            ad[0].data = ad[0].data.astype(np.float32)
            remove_single_length_dimension(ad)  # in-place

        return adinputs

    def _nonlinearity_coeffs(self, ad):
        coeffs = getattr(self.inst_adlookup.array_properties.get(ad.read_mode()),
                         'coeffs', None)
        return coeffs if ad.is_single else [coeffs] * len(ad)
