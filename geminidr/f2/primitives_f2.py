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
from .parameters_f2 import ParametersF2

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class F2(Gemini, NearIR):
    """
    This is the class containing all of the preprocessing primitives
    for the F2 level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set(["GEMINI", "F2"])

    def __init__(self, adinputs, **kwargs):
        super(F2, self).__init__(adinputs, **kwargs)
        self.inst_lookups = 'geminidr.f2.lookups'
        self.parameters = ParametersF2

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

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by "
                            "standardizeInstrumentHeaders".format(ad.filename))
                continue

            # Standardize the headers of the input AstroData object. Update the
            # keywords in the headers that are specific to FLAMINGOS-2.
            log.status("Updating keywords that are specific to FLAMINGOS-2")

            # Filter name (required for IRAF?)
            ad.phu.set('FILTER', ad.filter_name(stripID=True, pretty=True),
                       self.keyword_comments['FILTER'])

            # Pixel scale (CJS: I'm putting this in the extension too!)
            pixel_scale = ad.pixel_scale()
            ad.phu.set('PIXSCALE', pixel_scale, self.keyword_comments['PIXSCALE'])
            ad.hdr.set('PIXSCALE', pixel_scale, self.keyword_comments['PIXSCALE'])

            for desc in ('read_noise', 'gain', 'non_linear_level',
                         'saturation_level'):
                kw = ad._keyword_for(desc)
                ad.hdr.set(kw, getattr(ad, desc)()[0], self.keyword_comments[kw])
                try:
                    ad.phu.remove(kw)
                except (KeyError, AttributeError):
                    pass

            if 'SPECT' in ad.tags:
                kw = ad._keyword_for('dispersion_axis')
                self.hdr.set(kw, 2, self.keyword_comments(kw))

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=params["suffix"],
                                              strip=True)
        return adinputs

    def standardizeStructure(self, adinputs=None, **params):
        """
        This primitive is used to standardize the structure of GMOS data,
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
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        adoutputs = []
        for ad, mdf in zip(*gt.make_lists(adinputs, params['mdf'])):
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by standardizeStructure".
                            format(ad.filename))
                adoutputs.append(ad)
                continue

            # Attach an MDF to each input AstroData object
            if params["attach_mdf"]:
                ad = self.addMDF([ad], mdf=mdf)[0]

            # Raw FLAMINGOS-2 pixel data have three dimensions (2048x2048x1).
            # Remove the single length dimension from the pixel data.
            # CD3_3 keyword must also be removed or alignAndStack complains.
            ad = remove_single_length_dimension(ad)

            # Need to change dtype from int32 to float32, or else numpy will
            # promote to float64. There's no VAR or DQ at this stage.
            ad[0].data = ad[0].data.astype(np.float32)

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=params["suffix"],
                                              strip=True)
            adoutputs.append(ad)
        return adoutputs